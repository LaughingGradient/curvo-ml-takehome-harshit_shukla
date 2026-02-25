from datasets import load_dataset

def main():
    ds = load_dataset("roskoN/dailydialog")
    ds.cleanup_cache_files()
    print(ds)

if __name__ == "__main__":
    main()