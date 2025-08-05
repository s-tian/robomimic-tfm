import argparse
import h5py

def main():
    parser = argparse.ArgumentParser(description="Subset an HDF5 file by copying the first N demos to a new file.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input HDF5 file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output HDF5 file")
    parser.add_argument("--num-demos", "-n", type=int, required=True, help="Number of demos to copy")
    args = parser.parse_args()

    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
    NUM_DEMOS = args.num_demos

    with h5py.File(INPUT_PATH, 'r') as f_in, h5py.File(OUTPUT_PATH, 'w') as f_out:
        # --- Copy the first N demos under /data ---
        f_in_data = f_in['data']
        data_group = f_out.create_group("data")
        demo_names = sorted(list(f_in_data.keys()))
        selected_demos = demo_names[:NUM_DEMOS]
        print(f"Copying {len(selected_demos)} demos to {OUTPUT_PATH}")

        for demo_name in selected_demos:
            f_in_data.copy(demo_name, data_group)

        # --- Copy /data group attributes ---
        for attr_name, attr_value in f_in_data.attrs.items():
            data_group.attrs[attr_name] = attr_value

        # --- Copy /mask group if it exists, but only include subsetted demos ---
        if "mask" in f_in:
            mask_group_in = f_in["mask"]
            mask_group_out = f_out.create_group("mask")
            for filter_key in mask_group_in.keys():
                demo_names = [name.decode("utf-8") for name in mask_group_in[filter_key][()]]
                filtered_demos = [name for name in demo_names if name in selected_demos]
                if filtered_demos:
                    # Store as bytes, as in original
                    mask_group_out.create_dataset(filter_key, data=[n.encode("utf-8") for n in filtered_demos])

if __name__ == "__main__":
    main()