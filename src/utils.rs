use std::{ffi::{c_char, CString}, mem, ptr::NonNull};
use metal::*;

use objc2::rc::Retained;
use objc2_app_kit::{NSBackingStoreType, NSColor, NSScreen, NSWindow, NSWindowStyleMask};
use objc2_foundation::{CGPoint, MainThreadMarker, NSDate, NSRect, NSSize, NSString};


//Metal utils

pub fn new_render_pass_descriptor(texture : &TextureRef) -> &RenderPassDescriptorRef {

    let render_pass_descriptor = RenderPassDescriptor::new();
    let render_pass_attachment = render_pass_descriptor.color_attachments().object_at(0).unwrap();
    render_pass_attachment.set_texture(Some(&texture));
    render_pass_attachment.set_load_action(MTLLoadAction::Clear);
    render_pass_attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
    render_pass_attachment.set_store_action(MTLStoreAction::Store);

    render_pass_descriptor
}

pub fn new_metal_layer(device: &DeviceRef) -> MetalLayer {
    let layer = MetalLayer::new();

    layer.set_device(device);
    layer.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    layer.set_presents_with_transaction(false);

    return layer;
}

pub fn get_library(device : &DeviceRef) -> Library {
    let library_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/shaders.metallib");
    device.new_library_with_file(library_path).expect("Library not found")
}

pub fn prepare_pipeline_state (
    device : &DeviceRef,
    vertex_shader : &str,
    fragment_shader : &str,
    shaderlib : &Library
) -> RenderPipelineState {

    let vert = shaderlib.get_function(vertex_shader, None).expect("Could not find vertex function in library");
    let frag = shaderlib.get_function(fragment_shader, None).expect("Could not find fragment function in library");

    let pipeline_state_descriptor = RenderPipelineDescriptor::new();
    pipeline_state_descriptor.set_vertex_function(Some(&vert));
    pipeline_state_descriptor.set_fragment_function(Some(&frag));

    let pipeline_attachment = pipeline_state_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();

    pipeline_attachment.set_pixel_format(MTLPixelFormat::RGBA8Unorm);

    //can customize these pipeline attachments
    pipeline_attachment.set_blending_enabled(true);
    pipeline_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    pipeline_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    pipeline_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    pipeline_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    pipeline_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    pipeline_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);


    device
        .new_render_pipeline_state(&pipeline_state_descriptor)
        .unwrap()
}

pub fn make_buf<T>(data : &Vec<T>, device : &DeviceRef) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (mem::size_of::<T>() * data.len())as u64,
        MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeManaged
    )
}

pub fn copy_to_buf<T>(data : &Vec<T>, dst : &Buffer) {
    let buf_pointer = dst.contents(); //how does this grab a mut pointer from a non mutable reference?
    unsafe {
        std::ptr::copy(
            data.as_ptr(),
            buf_pointer as *mut T,
            data.len() as usize
        );
    }
    dst.did_modify_range(NSRange::new(
        0 as u64,
        (data.len() * size_of::<T>()) as u64
    ));
}

// pub fn prepare_compute_state(device: &DeviceRef) {
//     let descriptor = ComputePassDescriptor::new();
//     descriptor.set_dispatch_type(MTLDis);
// }

//AppKit utils

pub fn init_nsstring(str : &str, thread : MainThreadMarker) -> Retained<NSString> {
    let cstring = CString::new(str).expect("CString::new failed!");
    let ptr : NonNull<c_char> = NonNull::new(cstring.as_ptr() as *mut i8).expect("NonNull::new failed!");

    unsafe {
        NSString::initWithCString_encoding(
            thread.alloc::<NSString>(),
            ptr,
            NSString::defaultCStringEncoding()
        ).expect("String init failed!")
    }
}

pub fn initialize_window(width : f64, height : f64, color : (f64, f64, f64, f64), title : &str, style_mask : NSWindowStyleMask, thread : MainThreadMarker) -> Retained<NSWindow> {
    let screen_rect : NSRect = NSScreen::mainScreen(thread).unwrap().frame();
    let size  = NSSize{
        width,
        height
    };
    let origin= CGPoint{
        x: (screen_rect.size.width - width) * 0.5,
        y: (screen_rect.size.height - height) * 0.5,
    };
    let window_rect : NSRect = NSRect::new(
        origin,
        size
    );

    let window_color = unsafe {NSColor::colorWithSRGBRed_green_blue_alpha(color.0, color.1, color.2, color.3)};
    let window_title = init_nsstring(title, thread);

    let window  = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
        thread.alloc::<NSWindow>(),
        window_rect,
        style_mask,
        NSBackingStoreType::NSBackingStoreBuffered,
        false)
    };

    window.setBackgroundColor(Some(&window_color));
    window.setTitle(&window_title);
    window.contentView().unwrap().setWantsLayer(true);
    return window;
}

pub fn set_window_layer(window : &Retained<NSWindow>, layer: &MetalLayer) {
    unsafe {
        window.contentView().expect("Error setting window layer").setLayer(mem::transmute(layer.as_ref()));
    }
}

pub fn get_next_frame (fps : f64) -> Retained<NSDate> {
    unsafe {
        NSDate::dateWithTimeIntervalSinceNow(1.0 / fps)
    }
}
