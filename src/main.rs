mod utils;
mod maths;

use std::{ffi::c_float, mem};

use objc2::rc::Retained;
use rand::{random, seq::SliceRandom, thread_rng};
use objc::rc::autoreleasepool;
use objc2_app_kit::{NSAnyEventMask, NSApp, NSApplication, NSApplicationActivationPolicy, NSBitmapImageRep, NSEventType, NSImage, NSWindowStyleMask, NSCursor};
use objc2_foundation::{MainThreadMarker, NSComparisonResult, NSDate, NSDefaultRunLoopMode};
use utils::{copy_to_buf, get_library, get_next_frame, init_nsstring, initialize_window, make_buf, new_metal_layer, new_render_pass_descriptor, prepare_pipeline_state, set_window_layer};
use maths::{calculate_quaternion, float3_add, float3_subtract, quat_mult, scale2, scale3, update_quat_angle, Float2, Float3, Float4};

use metal::*;

#[repr(C)]
#[derive(Debug)]
enum PType {
    Grass,
    Fire,
    Water,
    Other
}

impl From<usize> for PType {
    fn from(value: usize) -> Self {
        match value {
            0 => PType::Grass,
            1 => PType::Fire,
            2 => PType::Water,
            3 => PType::Other,
            _ => PType::Other,
        }
    }
}

struct Particle {
    pos : Float2,
    ptype : PType,
}

#[repr(C)]
struct Args {
    group_width : u32,
    group_height : u32,
    step : u32,
    num : u32
}


fn main() {
    let mtm = MainThreadMarker::new().expect("Not running on main thread");
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);

    let style_mask =
        NSWindowStyleMask::Titled.union(
        NSWindowStyleMask::Closable);

    let view_width = 768.0;
    let view_height = 768.0;

    let window = initialize_window(view_width, view_height, (0.0, 0.0, 0.0, 1.0), "Falling Sand", style_mask, mtm);

    let device = Device::system_default().expect("Error getting GPU device");

    let layer = new_metal_layer(&device);
    set_window_layer(&window, &layer);

    unsafe {
        app.finishLaunching();
        app.activateIgnoringOtherApps(true);
        window.makeKeyAndOrderFront(None);
    }

    let num_particles = 2u32.pow(10) as usize;

    let mut particle_positions : Vec<Float3> = Vec::with_capacity(num_particles);
    let mut particle_velocities : Vec<Float3> = Vec::with_capacity(num_particles);
    let mut particle_delta_vs : Vec<Float3> = Vec::with_capacity(num_particles);
    let mut spatial_indices : Vec<(u32, u32, u32, u32)> = Vec::with_capacity(num_particles);
    let mut spatial_offsets : Vec<u32> = Vec::with_capacity(num_particles);
    let mut materials : Vec<PType> = Vec::with_capacity(num_particles);
    let mut cells : Vec<u32> = Vec::with_capacity(num_particles);


    for i in 0..num_particles {
        // particle_positions.push(Float2((i as f32 / num_particles as f32) * 2.0 - 1.0, (i as f32 / num_particles as f32) * 2.0 - 1.0));
        particle_positions.push(Float3(random::<f32>(), random::<f32>(), random::<f32>() + 1.0));
        particle_velocities.push(Float3(random::<f32>() / 2.0 - 0.25, random::<f32>() / 10.0 - 0.05, random::<f32>() / 10.0 - 0.05));
        particle_delta_vs.push(particle_velocities[i]);
        spatial_indices.push((0, 0, 0, 0));
        spatial_offsets.push(num_particles as u32);
        materials.push((i % 3).into());
        cells.push(0);
    }

    println!("{:?}", particle_positions);
    /*
    specification of the system:
        handle addition and removal of particles from the simulation
        handle physics between particles
        handle specific interactions between particle types on collision
            these are probably structured interactions that only occur on collision
            if we want interactions that can happen at a distance, probably add particle
            that carries that interaction to next particles
            e.g. if fire and water collide, remove fire particle and add steam particle

        new consideration, particles have different baseline physics updates based on particle type
        how to we distinguish between them in the sim?

        operate on all particles and match sim logic to particle type?
            many branches, but might be cheap enough and worth the reduced abstraction
            physics math should be relatively fast, switching between mult types might not be problematic


    */


    let shader_lib = get_library(&device);

    let particle_pipeline = prepare_pipeline_state(
        &device,
        "rect_vertex",
        "rect_fragment",
        &shader_lib
    );

    //write a compute shader that updates the positions on the particles
    //handle only gravity initially

    let physics_function = shader_lib.get_function("physics_kernel", None).expect("err finding physics function");
    let physics_pipeline = device.new_compute_pipeline_state_with_function(&physics_function).expect("Error creating pipeline");


    let threads = num_particles as u64;
    let threads_per_grid = MTLSize::new(threads, 1, 1);
    let threads_per_threadgroup = MTLSize::new(threads.min(physics_pipeline.max_total_threads_per_threadgroup()), 1, 1);




    //write a compute shader that handles forces between particles
    //maybe use spatial hashing in order to cut interaction counts
    let hash_function = shader_lib.get_function("hash_kernel", None).expect("err finding hash function");
    let hash_pipeline = device.new_compute_pipeline_state_with_function(&hash_function).expect("Error creating pipeline");

    let sort_function = shader_lib.get_function("sort_kernel", None).expect("err finding sort function");
    let sort_pipeline = device.new_compute_pipeline_state_with_function(&sort_function).expect("Error creating pipeline");

    let sort_threads_per_grid = MTLSize::new(threads / 2, 1, 1);
    let sort_threads_per_threadgroup = MTLSize::new(threads.min(physics_pipeline.max_total_threads_per_threadgroup()), 1, 1);

    let offset_function = shader_lib.get_function("offset_kernel", None).expect("err finding offset function");
    let offset_pipeline = device.new_compute_pipeline_state_with_function(&offset_function).expect("Error creating pipeline");

    let collision_function = shader_lib.get_function("collision_kernel", None).expect("err finding collision function");
    let collision_pipeline = device.new_compute_pipeline_state_with_function(&collision_function).expect("Error creating pipeline");

    let update_function = shader_lib.get_function("update_kernel", None).expect("err finding update function");
    let update_pipeline = device.new_compute_pipeline_state_with_function(&update_function).expect("Error creating pipeline");

    //write a compute shader that handles swapping particle type based on interactions



    let position_buf = make_buf(&particle_positions, &device);
    let velocity_buf = make_buf(&particle_velocities, &device);
    let delta_buf = make_buf(&particle_delta_vs, &device);

    let spatial_index_buf = make_buf(&spatial_indices, &device);
    let spatial_offsets_buf = make_buf(&spatial_offsets, &device);

    let materials_buf = make_buf(&materials, &device);
    let cells_buf = make_buf(&cells, &device);

    let gravity : f32 = -0.1;
    let delta_time : f32 = 1.0 / 30.0;

    let cell_size : f32 = 0.25;
    let cell_count = (2.0 / cell_size).powf(3.0);
    // for _ in 0..cell_count as u32 {
    //     spatial_offsets.push(num_particles as u32);
    // }

    let offset_threads_per_grid = MTLSize::new(cell_count as u64, 1, 1);
    let offset_threads_per_threadgroup = MTLSize::new((cell_count as u64).min(physics_pipeline.max_total_threads_per_threadgroup()), 1, 1);
    let power_of_two = (num_particles as f32).log2() as u32;

    loop {
        autoreleasepool(|| {
            if app.windows().is_empty() {
                unsafe {app.terminate(None)};
            }

            let drawable = layer.next_drawable().expect("err getting drawable");
            let texture = drawable.texture();
            let particle_descriptor = new_render_pass_descriptor(texture);

            let command_queue = device.new_command_queue();

            let command_buffer = command_queue.new_command_buffer();
            let physics_encoder = command_buffer.new_compute_command_encoder();
            physics_encoder.set_compute_pipeline_state(&physics_pipeline);
            physics_encoder.set_buffer(0, Some(&position_buf), 0);
            physics_encoder.set_buffer(1, Some(&delta_buf), 0);
            physics_encoder.set_bytes(2, size_of::<f32>() as u64, vec![gravity].as_ptr() as *const _);
            physics_encoder.set_bytes(3, size_of::<f32>() as u64, vec![delta_time].as_ptr() as *const _);
            physics_encoder.set_bytes(4, size_of::<u32>() as u64, vec![num_particles as u32].as_ptr() as *const _);

            physics_encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            physics_encoder.end_encoding();

            let hash_encoder = command_buffer.new_compute_command_encoder();
            hash_encoder.set_compute_pipeline_state(&hash_pipeline);
            hash_encoder.set_buffer(0, Some(&position_buf), 0);
            hash_encoder.set_buffer(1, Some(&spatial_index_buf), 0);
            hash_encoder.set_buffer(2, Some(&spatial_offsets_buf), 0);
            hash_encoder.set_bytes(3, size_of::<f32>() as u64 * 2, vec![cell_size, cell_count].as_ptr() as *const _);

            hash_encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            hash_encoder.end_encoding();

            let sort_encoder = command_buffer.new_compute_command_encoder();
            sort_encoder.set_compute_pipeline_state(&sort_pipeline);
            sort_encoder.set_buffer(0, Some(&spatial_index_buf), 0);
            for stage in 0..power_of_two {
                for step in 0..(stage + 1) {
                    let group_width = 1 << (stage - step);
                    let group_height = 2 * group_width;
                    let args = Args {
                        group_width,
                        group_height,
                        step,
                        num : num_particles as u32,
                    };
                    sort_encoder.set_bytes(1, size_of::<u32>() as u64 * 4, vec![args].as_ptr() as *const _);
                    sort_encoder.dispatch_threads(sort_threads_per_grid, sort_threads_per_threadgroup);
                }
            }
            sort_encoder.end_encoding();

            let offset_encoder = command_buffer.new_compute_command_encoder();
            offset_encoder.set_compute_pipeline_state(&offset_pipeline);
            offset_encoder.set_buffers(0, &[Some(&spatial_index_buf), Some(&spatial_offsets_buf)], &[0, 0]);
            offset_encoder.set_bytes(2, size_of::<u32>() as u64, vec![num_particles as u32].as_ptr() as *const _);
            offset_encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            offset_encoder.end_encoding();

            let collision_encoder = command_buffer.new_compute_command_encoder();
            collision_encoder.set_compute_pipeline_state(&collision_pipeline);
            collision_encoder.set_buffers(0, &[Some(&position_buf), Some(&velocity_buf), Some(&delta_buf), Some(&materials_buf), Some(&spatial_index_buf), Some(&spatial_offsets_buf)], &[0, 0, 0, 0, 0, 0]);
            collision_encoder.set_bytes(6, size_of::<f32>() as u64 * 2, vec![cell_size, cell_count].as_ptr() as *const _);
            collision_encoder.set_bytes(7, size_of::<f32>() as u64, vec![num_particles as f32].as_ptr() as *const _);
            collision_encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            collision_encoder.end_encoding();

            let update_encoder = command_buffer.new_compute_command_encoder();
            update_encoder.set_compute_pipeline_state(&update_pipeline);
            update_encoder.set_buffers(0, &[Some(&velocity_buf), Some(&delta_buf)], &[0, 0]);
            update_encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            update_encoder.end_encoding();

            let particle_encoder = command_buffer.new_render_command_encoder(&particle_descriptor);
            particle_encoder.set_render_pipeline_state(&particle_pipeline);
            particle_encoder.set_vertex_buffer(0, Some(&position_buf), 0);
            particle_encoder.set_vertex_buffer(1, Some(&velocity_buf), 0);
            particle_encoder.set_vertex_buffer(2, Some(&materials_buf), 0);
            particle_encoder.set_vertex_bytes(3, size_of::<f32>() as u64, vec![view_width as f32].as_ptr() as *const _);

            particle_encoder.draw_primitives_instanced(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                particle_positions.len() as u64
            );
            particle_encoder.end_encoding();

            command_buffer.present_drawable(&drawable);
            command_buffer.commit();

            loop {
                let event = unsafe {app.nextEventMatchingMask_untilDate_inMode_dequeue(
                    NSAnyEventMask,
                    None,
                    NSDefaultRunLoopMode,
                    true
                )};
                match event {
                    Some(ref e) => {
                        unsafe {
                            match e.r#type() {
                                _ => {}
                            }
                            app.sendEvent(&e);
                        }
                    },
                    None => break,
                }
            }
            command_buffer.wait_until_completed();
            // let sorted : &[(u32,u32,u32,u32)] = unsafe{std::slice::from_raw_parts(spatial_index_buf.contents().cast(), num_particles)};
            // let offsets : &[u32] = unsafe{std::slice::from_raw_parts(spatial_offsets_buf.contents().cast(), cell_count as usize)};
            // let cells : &[u32] = unsafe{std::slice::from_raw_parts(cells_buf.contents().cast(), cell_count as usize)};
            // println!("sorted list: {sorted:?}");
            // println!("offsets: {:?}", offsets);
            // println!("cells: {:?}", cells);
            // let mut new_index = 0;
            // println!("Test");
            // for i in 0..sorted.len() - 1 {
            //     if sorted[i].0 > sorted[i + 1].0 {
            //         print!("{}, ", (i + 1) - new_index);
            //         new_index = i + 1;
            //     }
            // }
            // println!();
        });
        // break;
    }
}
