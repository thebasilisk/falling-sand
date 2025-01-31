#include <metal_stdlib>

using namespace metal;

static constant uint hash1 = 15823;
static constant uint hash2 = 9737333;

static constant float max_velocity = 0.5;

int2 cell_from_position (float2 pos, float radius) {
    return (int2)floor(pos / radius);
}

uint hash_cell (int2 pos) {
    uint2 upos = (uint2)pos;
    uint a = hash1 * upos.x;
    uint b = hash2 * upos.y;
    return a + b;
}

uint key_from_hash (uint hash, uint count) {
    return hash % count;
}


kernel void hash_kernel (
    const device float2 *pos [[ buffer(0) ]],
    device uint4 *spatial_indices [[ buffer(1) ]],
    device uint *spatial_offsets [[ buffer(2) ]],
    const device float *cell_size_count [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    int2 cell = cell_from_position(pos[gid], cell_size_count[0]);
    uint hash = hash_cell(cell);
    spatial_indices[gid] = uint4(key_from_hash(hash, (uint)cell_size_count[1]), hash, gid, 0);
    spatial_offsets[min(gid, (uint)cell_size_count[1])] = (uint)cell_size_count[1];
}

struct sort_args {
    uint group_width;
    uint group_height;
    uint step_index;
    uint num_entries;
};

kernel void sort_kernel (
    device uint4 *spatial_indices [[ buffer(0) ]],
    device sort_args *args [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    auto group_width = args->group_width;
    auto group_height = args->group_height;
    auto step_index = args->step_index;
    auto num_entries = args->num_entries;

    uint index_in_group = gid & (group_width-1);
    uint left_index = index_in_group + group_height * (gid / group_width);
    uint step = step_index == 0 ? (group_height - 1) - (2 * index_in_group) : group_height / 2;
    uint right_index = left_index + step;

    if (right_index > num_entries) {
        return;
    }

    uint value_left = spatial_indices[left_index].x;
    uint value_right = spatial_indices[right_index].x;

    if (value_left > value_right) {
        uint4 temp = spatial_indices[left_index];
        spatial_indices[left_index] = spatial_indices[right_index];
        spatial_indices[right_index] = temp;
    }
}

kernel void offset_kernel (
    device uint4 *spatial_indices [[ buffer(0) ]],
    device uint *spatial_offsets [[ buffer(1) ]],
    device uint *num [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid > num[0]) return;

    uint end = num[0];

    uint key = spatial_indices[gid].x;
    uint prev = gid == 0 ? end : spatial_indices[gid - 1].x;

    if (key != prev) {
        spatial_offsets[key] = gid;
    }
}


kernel void physics_kernel (
    device float2 *pos [[ buffer(0) ]],
    device float2 *vels [[ buffer(1) ]],
    const device float *grav [[ buffer(2) ]],
    const device float *del_t [[ buffer(3) ]],
    const device uint *len [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid > len[0]) return;
    vels[gid] += float2(0.0, grav[0]) * del_t[0];
    if (length(vels[gid]) >= max_velocity) {
        vels[gid] -= float2(0.0, grav[0]) * del_t[0];
    }
    pos[gid] += vels[gid] * del_t[0];

    if (pos[gid].y <= 0.0) {
        pos[gid] = float2(pos[gid].x, -0.0);
        vels[gid] = float2(vels[gid].x, vels[gid].y * -1);
    }
    if (pos[gid].x <= 0.0 || pos[gid].x >= 1.0) {
        vels[gid] = float2(vels[gid].x * -1, vels[gid].y);
    }
}

enum PType {
    Grass,
    Fire,
    Water,
    Other
};

static constant int2 grid_offsets[9] = {
    int2(-1, 1),
	int2(0, 1),
	int2(1, 1),
	int2(-1, 0),
	int2(0, 0),
	int2(1, 0),
	int2(-1, -1),
	int2(0, -1),
	int2(1, -1),
};

kernel void collision_kernel (
    device float2 *pos [[ buffer(0) ]],
    device float2 *vels [[ buffer(1) ]],
    device PType *mats [[ buffer(2) ]],
    const device uint4 *spatial_indices [[ buffer(3) ]],
    const device uint *spatial_offsets [[ buffer(4) ]],
    const device float *cell_size_count [[ buffer(5) ]],
    const device float *num [[ buffer(6) ]],
    device uint *offsets [[ buffer(7) ]],
    uint gid [[ thread_position_in_grid ]]
) {

    float2 position = pos[gid];
    float threshold = 0.01;

    for (int n = 0; n < 9; n++) {
        uint hash = hash_cell(cell_from_position(position, cell_size_count[0]) + grid_offsets[n]);
        uint key = key_from_hash(hash, (uint)cell_size_count[1]);
        uint offset = spatial_offsets[key];
        if (gid <= cell_size_count[1]) {
            offsets[gid] = offset;
        }

        uint i = 0;
        uint4 current = spatial_indices[offset];
        while (offset + i < 1024) {
            if (current.x != key) break;
            if (current.y != hash) break;
            if (current.z == gid) {
                i++;
                current = spatial_indices[offset + i];
                continue;
            }

            if (distance(pos[current.z], position) <= threshold) {
                float2 temp = vels[current.z];
                vels[current.z] = vels[gid];
                vels[gid] = temp;

                if (mats[current.z] != mats[gid]) {
                    if (mats[current.z] == Grass) {
                        if (mats[gid] == Fire) {
                            mats[current.z] = Fire;
                        } else {
                            mats[gid] = Grass;
                        }
                    }
                    if (mats[current.z] == Fire) {
                        if (mats[gid] == Water) {
                            mats[current.z] = Water;
                        } else {
                            mats[gid] = Fire;
                        }
                    }
                    if (mats[current.z] == Water) {
                        if (mats[gid] == Grass) {
                            mats[current.z] = Grass;
                        } else {
                            mats[gid] = Water;
                        }
                    }
                    break;
                }
            }
            i++;
            current = spatial_indices[offset + i];
        }
    }
}


struct ColorInOut {
    float4 position [[ position ]];
    float4 color;
};

vertex ColorInOut rect_vertex (
    const device float2 *positions [[ buffer(0) ]],
    const device float2 *velocities [[ buffer(1) ]],
    const device PType *materials [[ buffer(2) ]],
    const device float *view_width [[ buffer(3) ]],
    uint vid [[ vertex_id ]],
    uint id [[ instance_id ]]
) {
    ColorInOut out;

    auto device const &pos = positions[id];
    auto device const &vel = velocities[id];

    int vid_bit1 = vid % 2;
    int vid_bit2 = vid / 2;
    float size = 0.005;

    float2 normalized_pos = pos * 2.0 - float2(1.0, 1.0);

    float x = normalized_pos.x + (size) * (2 * vid_bit1 - 1);
    float y = normalized_pos.y - (size) * (2 * vid_bit2 - 1);


    float4 color_excited;
    switch (materials[id]) {
        case 0 : color_excited = float4(0.0, 1.0, 0.0, 1.0); break;
        case 1 : color_excited = float4(1.0, 0.0, 0.0, 1.0); break;
        case 2 : color_excited = float4(0.0, 0.0, 1.0, 1.0); break;
        case 3 : color_excited = float4(0.5, 0.5, 0.5, 1.0); break;
    }
    float4 color_base = color_excited * 0.5;

    out.position = float4(x, y, 0.0, 1.0);
    out.color = mix(color_base, color_excited, length(vel) / max_velocity);
    //out.color = color_excited;

    return out;
}

fragment float4 rect_fragment (
    ColorInOut in [[ stage_in ]]
) {
    return in.color;
}
