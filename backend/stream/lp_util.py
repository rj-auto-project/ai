import string
import easyocr
reader = easyocr.Reader(["en"], gpu=False)

dict_char_to_int = {
    "O": "0",
    "I": "1",
    "J": "3",
    "A": "4",
    "G": "6",
    "S": "5",
    "B": "8",
    "Z": "7",
    "T": "1",
    "@": "0"
}

dict_int_to_char = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "J",
    "4": "A",
    "6": "G",
    "5": "S",
    "6": "",
    "8": "B",
    "@": "D"
}

def remove_special_chars(s, chars="!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/~` "):
    return s.strip(chars)

def license_complies_format(text):
    if (
        (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys())
        and (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys())
        and (
            text[2] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[2] in dict_char_to_int.keys()
        )
        # and (
        #     text[3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        #     or text[3] in dict_char_to_int.keys()
        # )
        # and (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys())
        and (text[-5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys())
        and (
            text[-4] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[-4] in dict_char_to_int.keys()
        )
        and (
            text[-3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[-3] in dict_char_to_int.keys()
        )
        and (
            text[-2] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[-2] in dict_char_to_int.keys()
        )
        and (
            text[-1] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[-1] in dict_char_to_int.keys()
        )
    ):
        return True
    else:
        return False

def format_license(text):
    license_plate_ = ""
    mapping = {
        0: dict_int_to_char,
        1: dict_int_to_char,
        2: dict_char_to_int,
        3: dict_char_to_int,
        4: dict_int_to_char,
        5: dict_int_to_char,
        -4: dict_char_to_int,
        -3: dict_char_to_int,
        -2: dict_char_to_int,
        -1: dict_char_to_int,
    }
    for j in [0, 1, 2, -4, -3, -2, -1]:
        if text[j] in mapping[j].keys():

            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_

def read_license_plate(license_plate_crop :str, camera_ip :str, time :str):
    detections = reader.readtext(license_plate_crop)
    print(detections,"----")
    for i, detection in enumerate(detections):
        bbox, text, score = detection

        text = text.upper().replace(" ", "")
        text = remove_special_chars(text)
        text = text.replace("\n", "")
        text = text.upper().replace(" ", "")
        print(text)
        format_judge = False
        if license_complies_format(text):
            format_judge = True
        with open(
                "/home/annone/ai/backend/stream/lp_logs.csv",
                "a",
            ) as f:
            f.write(
                    "{},{},{}\n".format(camera_ip, text, time)
                )
            f.close()
        print(text,score)
        return text, score