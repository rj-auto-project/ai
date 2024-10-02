from google_images_download import google_images_download

# Create an instance of google_images_download
response = google_images_download.googleimagesdownload()

# Define search arguments
arguments = {
    "keywords": "ashirvad atta",
    "limit": 20,  # Number of images to download
    "print_urls": True,
    "output_directory": "./images"
}

# Download the images
response.download(arguments)
