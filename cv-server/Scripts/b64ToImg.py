import base64

# Function to convert base64 string to image file
def base64_to_image(base64_string, output_file):
    img_data = base64.b64decode(base64_string)
    with open(output_file, 'wb') as f:
        f.write(img_data)

# Example usage
base64_string = "your_base64_string_here"
output_file = "output_image.png"
base64_to_image(base64_string, output_file)