# create_label_mapping.py
import json
import os


def create_label_mapping():
    # Chỉ tạo mapping cho các chữ cái đã có
    label_mapping = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'J',
        4: 'Z'
    }

    # Lưu vào file JSON
    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)

    print("Created label_mapping.json with current letters")
    print(f"Mapping: {label_mapping}")


if __name__ == "__main__":
    create_label_mapping()