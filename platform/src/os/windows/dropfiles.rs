use crate::{
    event::DragItem,
    live_id::LiveId,
    log,
    windows::Win32::{
        Foundation::HGLOBAL,
        System::{
            Com::STGMEDIUM,
            Memory::{
                GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock, GMEM_FIXED, GMEM_ZEROINIT,
            },
            Ole::ReleaseStgMedium,
        },
    },
};

// This is where all binary conversion code goes between makepad DragItem and Windows HGLOBAL/DROPFILES structure

// convert incoming STGMEDIUM from internal or external source to DragItem
pub fn convert_medium_to_dragitem(medium: STGMEDIUM) -> Option<DragItem> {
    // get size and raw pointer
    let hglobal_size = unsafe { GlobalSize(medium.u.hGlobal) };
    let hglobal_raw_ptr = unsafe { GlobalLock(medium.u.hGlobal) };

    // debug dump
    /*
    let u8_slice = unsafe { std::slice::from_raw_parts_mut(hglobal_raw_ptr as *mut u8,hglobal_size) };
    for i in 0..(hglobal_size >> 4) {
        let mut line = String::new();
        for k in 0..16 {
            line.push_str(&format!(" {:02X}",u8_slice[(i << 4) + k]));
        }
        log!("{:04X}: {}",i << 4,line);
    }
    if (hglobal_size & 15) != 0 {
        let mut line = String::new();
        for k in 0..(hglobal_size & 15) {
            line.push_str(&format!(" {:02X}",u8_slice[((hglobal_size >> 4) << 4) + k]));
        }
        log!("{:04X}: {}",(hglobal_size >> 4) << 4,line);
    }
    */

    // read DROPFILES part
    let u32_slice = unsafe { std::slice::from_raw_parts_mut(hglobal_raw_ptr as *mut u32, 7) };
    let names_offset = u32_slice[0];
    let has_wide_strings = u32_slice[4];

    // guard against non-wide strings or unknown objects
    if has_wide_strings == 0 {
        log!("drag object should have wide strings");
        let _ = unsafe { GlobalUnlock(medium.u.hGlobal) };
        unsafe { ReleaseStgMedium(&medium as *const STGMEDIUM as *mut STGMEDIUM) };
        return None;
    }

    if (names_offset != 20) && (names_offset != 28) {
        log!("unknown drag object");
        let _ = unsafe { GlobalUnlock(medium.u.hGlobal) };
        unsafe { ReleaseStgMedium(&medium as *const STGMEDIUM as *mut STGMEDIUM) };
        return None;
    }

    let mut internal_id: Option<LiveId> = None;
    let u16_slice = if names_offset == 20 {
        // regular DROPFILES from external source
        unsafe {
            std::slice::from_raw_parts_mut(
                (hglobal_raw_ptr as *mut u8).offset(20) as *mut u16,
                (hglobal_size - 20) / 2,
            )
        }
    } else {
        let id = LiveId(((u32_slice[6] as u64) << 32) | (u32_slice[5] as u64));
        if id.0 != 0 {
            internal_id = Some(id);
        }
        // internal DROPFILES with internal ID as well
        //let u64_slice = unsafe { std::slice::from_raw_parts_mut((hglobal_raw_ptr as *mut u8).offset(20) as *mut u64,1) };
        //internal_id = Some(LiveId(u64_slice[0]));
        unsafe {
            std::slice::from_raw_parts_mut(
                (hglobal_raw_ptr as *mut u8).offset(28) as *mut u16,
                (hglobal_size - 28) / 2,
            )
        }
    };

    // extract/decode filenames
    let mut filenames = Vec::<String>::new();
    let mut filename = String::new();
    for w in u16_slice {
        if *w != 0 {
            let c = match char::from_u32(*w as u32) {
                Some(c) => c,
                None => char::REPLACEMENT_CHARACTER,
            };
            filename.push(c);
        } else {
            if (filename.len() == 0) && (filenames.len() > 0) {
                break;
            }
            filenames.push(filename);
            filename = String::new();
        }
    }

    /*
    for filename in filenames.iter() {
        log!("    \"{}\"",filename);
    }*/

    // ready
    let _ = unsafe { GlobalUnlock(medium.u.hGlobal) };
    unsafe { ReleaseStgMedium(&medium as *const STGMEDIUM as *mut STGMEDIUM) };

    if filenames.is_empty() {
        return None;
    }

    // All of them. The loop above already parses every name out of CF_HDROP —
    // the previous code threw the whole batch away whenever more than one
    // arrived, which is precisely the common case: selecting several files and
    // dragging them in as a group.
    Some(DragItem::FilePath {
        paths: filenames,
        internal_id,
    })
}

// create new internal DROPFILES structure from DragItem
pub fn create_hglobal_for_dragitem(drag_item: &DragItem) -> Option<HGLOBAL> {
    if let DragItem::FilePath { paths, internal_id } = drag_item {
        if paths.is_empty() {
            return None;
        }

        // Encode every filename. CF_HDROP is a double-null-terminated list:
        // "a\0b\0c\0\0", so several files go out in one drag just as they come
        // in — same format, both directions.
        let mut encoded_filename: Vec<u16> = Vec::new();
        for path in paths {
            encoded_filename.extend(path.encode_utf16());
            encoded_filename.push(0);
        }

        // final terminator of the list
        encoded_filename.push(0);

        // create HGLOBAL to contain DROPFILES structure, the internal ID and this encoded filename
        let size_in_bytes = 28 + encoded_filename.len() * 2;
        let hglobal = unsafe { GlobalAlloc(GMEM_ZEROINIT | GMEM_FIXED, size_in_bytes) }.unwrap();
        let hglobal_raw_ptr = unsafe { GlobalLock(hglobal) };

        // initialize DROPFILES part
        let u32_slice = unsafe { std::slice::from_raw_parts_mut(hglobal_raw_ptr as *mut u32, 7) };
        u32_slice[0] = 28; // offset to filename
        u32_slice[1] = 0;
        u32_slice[2] = 0;
        u32_slice[3] = 0;
        u32_slice[4] = 1; // not 0 because 16-bit characters in the filename

        // initialize internal ID
        if let Some(internal_id) = internal_id {
            u32_slice[6] = (internal_id.0 >> 32) as u32;
            u32_slice[5] = (internal_id.0 & 0xffff_ffff) as u32;
            //let u64_slice = unsafe { std::slice::from_raw_parts_mut((hglobal_raw_ptr as *mut u8).offset(20) as *mut u64,1) };
            //u64_slice[0] = internal_id.0;
        }

        // initialize filename
        unsafe {
            std::ptr::copy_nonoverlapping(
                encoded_filename.as_ptr(),
                (hglobal_raw_ptr as *mut u8).offset(28) as *mut u16,
                encoded_filename.len(),
            )
        };

        // ready
        unsafe { GlobalUnlock(hglobal) }.unwrap();

        Some(hglobal)
    } else {
        log!("only DragItem::FilePath supported");
        None
    }
}
