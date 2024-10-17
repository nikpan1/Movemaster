import base64

# Open the image file in binary mode
with open('SavedImage.png', 'rb') as image_file:
    # Read the image file
    image_data = image_file.read()
    # Encode the image data to Base64
    base64_encoded_data = base64.b64encode(image_data)
    # Convert the Base64 bytes to a string
    base64_string = base64_encoded_data.decode('utf-8')

print(base64_string)