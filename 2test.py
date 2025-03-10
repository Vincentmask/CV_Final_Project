import os

test_dataset_path = "./asl_alphabet_test/asl_alphabet_test"  # Update if needed

if not os.path.exists(test_dataset_path):
    print(f"❌ ERROR: The folder '{test_dataset_path}' does not exist.")
else:
    classes = os.listdir(test_dataset_path)
    print("✅ Found classes:", classes)
    for class_folder in classes:
        class_path = os.path.join(test_dataset_path, class_folder)
        if os.path.isdir(class_path):
            print(f"📂 {class_folder}: {len(os.listdir(class_path))} images")
