import cv2
import os
from mouth import mouth

expressions_dict = {
    "0": "neutral",
    "1": "pouting",
    "2": "smile",
    "3": "open_mouth",
    "-1": "error/undetectable"
}

def detect_expression_for_image(img_path):
    expression = mouth(img_path)
    detected_expression = expressions_dict.get(expression, "unknown")
    return detected_expression

def main():
    # Detect expression for each sequentially numbered image until an image is not found
    results = []
    index = 0
    while True:
        image_path = os.path.join("input_images", f"{index}.jpg")
        if not os.path.exists(image_path):
            break  # exit loop if we reach a number where the image doesn't exist
        
        detected_expression = detect_expression_for_image(image_path)
        results.append(f"{index}.jpg: {detected_expression}")
        index += 1

    # Save results to a text file
    with open("expression_results.txt", "w") as f:
        for line in results:
            f.write(line + "\n")

if __name__ == "__main__":
    main()
