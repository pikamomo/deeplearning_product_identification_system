# Import necessary libraries
from flask import Flask, request, json, render_template, jsonify  # Flask Web framework related
import torch  # PyTorch deep learning framework
import numpy as np  # Numerical computing library
from io import BytesIO
import base64
from image_denoising import denoising_config
from image_denoising import denoising_model
from image_classification import classification_config
from image_classification import classification_model
from image_similarity import similarity_config  # Configuration file
from image_similarity import similarity_model  # Custom model module
from flask import send_from_directory

# Import image processing and similarity calculation related libraries
from sklearn.neighbors import NearestNeighbors  # K-nearest neighbors algorithm
import torchvision.transforms as T  # Image preprocessing tools
import os  # Operating system interface library
from PIL import Image  # PIL image processing library


# Create Flask application instance, set static folder as 'dataset'
app = Flask(__name__, static_folder='../common/dataset')

@app.route('/pictures/<filename>')
def serve_pictures(filename):
    return send_from_directory('./pictures', filename)

# Print startup information
print("Starting application")

# Device detection and setup (prioritize GPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading denoising model")
denoiser = denoising_model.ConvDenoiser()
denoiser.load_state_dict(torch.load(
    os.path.join('../image_denoising', denoising_config.DENOISER_MODEL_NAME), map_location=device))
denoiser.to(device)
print("Denoising model loaded")

print("Loading classification model")
classifier = classification_model.Classifier()
classifier.load_state_dict(torch.load(
    os.path.join('../image_classification', classification_config.CLASSIFIER_MODEL_NAME), map_location=device))
classifier.to(device)
print("Classification model loaded")

# Load model before starting server
print("Loading embedding model")
encoder = similarity_model.ConvEncoder()  # Initialize encoder
# Load encoder's pretrained weights (automatically handles device mapping)
encoder.load_state_dict(
    torch.load(
        os.path.join(
            '..',
            similarity_config.PACKAGE_NAME, similarity_config.ENCODER_MODEL_NAME),
        map_location=device))
encoder.to(device)  # Move model to specified device
print("Embedding model loaded")

print("Loading vector database")
# Load pre-stored embedding matrix
embedding = np.load(os.path.join(
    '..',
    similarity_config.PACKAGE_NAME,
    similarity_config.EMBEDDING_NAME)
)
print("Vector database loaded")

def compute_similar_images(image_tensor, num_images, embedding, device):
    """
    Given an image and the number of similar images to generate.
    Returns a list of num_images most similar images

    Parameters:
    - image_tensor: tensor converted from image via PIL, need to find images similar to image_tensor.
    - num_images: number of similar images to find.
    - embedding: a tuple (num_images, embedding_dim), image embeddings learned from autoencoder.
    - device: "cuda" or "cpu" device.
    """

    image_tensor = image_tensor.to(device)  # Move image tensor to specified device

    with torch.no_grad():  # Disable gradient calculation
        # Generate image embedding representation through encoder
        image_embedding = encoder(image_tensor).cpu().detach().numpy()

    # Flatten embedding to 2D (number of samples x feature dimension)
    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))

    # Use KNN algorithm to find nearest neighbor images
    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)  # Fit on pre-stored embedding matrix

    # Execute KNN query (returns distances and indices)
    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()  # Convert to Python list format
    return indices_list


# Home page route
@app.route("/")
def index():
    # Render home page template
    return render_template('index.html')


# Example route: return JSON containing all image data
@app.route('/denoising', methods=['POST'])
def get_denoised_image():
    # Get image file from request
    image = request.files["image"]
    # Open image and convert to PIL format
    image = Image.open(image.stream).convert("RGB")
    # Define image preprocessing pipeline
    t = T.Compose([T.Resize((68, 68)), T.ToTensor()])
    # Apply preprocessing and convert to tensor
    image_tensor = t(image)

    ## Add random noise to input image
    # Generate random noise with same shape as tensor_image, multiply by noise_factor
    noisy_img = image_tensor + denoising_config.NOISE_FACTOR * torch.randn(*image_tensor.shape)
    # Clip image pixel values to [0, 1] range to avoid exceeding valid range
    noisy_img = torch.clip(noisy_img, 0., 1.)

    # Add batch dimension
    noisy_img = noisy_img.unsqueeze(0)


    with torch.no_grad():
        # Model inference
        noisy_img = noisy_img.to(device)
        denoised_image = denoiser(noisy_img)

    # Post-processing
    denoised_image = denoised_image.squeeze(0).cpu()  # Remove batch dimension
    denoised_image = denoised_image.permute(1, 2, 0).numpy() * 255  # CHW -> HWC and convert to 0-255 range
    noisy_img = noisy_img.squeeze(0).cpu()
    noisy_img = noisy_img.permute(1, 2, 0).numpy() * 255

    # denoised_image = np.moveaxis(denoised_image.detach().cpu().numpy(), 1, -1)
    # print("denoised_image shape: ", denoised_image.shape)
    #
    # plt.imshow(denoised_image[0])
    # plt.show()

    # Convert to PIL image
    denoised_image = Image.fromarray(denoised_image.astype('uint8'))
    noisy_img = Image.fromarray(noisy_img.astype('uint8'))

    def encode_image(img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    return (
        json.dumps(
            {
                "noisy_img": encode_image(noisy_img),
                "denoised_image": encode_image(denoised_image)
            }),
        200,
        {"ContentType": "application/json"},
    )

@app.route("/classification", methods=["POST"])
def classification():
    # Get image file from request
    image = request.files["image"]
    # Open image and convert to PIL format
    image = Image.open(image.stream).convert("RGB")
    # Define image preprocessing pipeline
    t = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    # Apply preprocessing and convert to tensor
    image_tensor = t(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    # Model inference
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        classification = classifier(image_tensor)

    return "The product type you searched for is: " + classification_config.classification_names[np.argmax(classification.cpu().detach().numpy())]

# Similar image calculation route (POST request)
@app.route("/simimages", methods=["POST"])
def simimages():
    # Get image file from request
    image = request.files["image"]
    # Open image and convert to PIL format
    image = Image.open(image.stream).convert("RGB")
    # Define image preprocessing pipeline
    t = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    # Apply preprocessing and convert to tensor
    image_tensor = t(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    # Calculate similar image indices
    indices_list = compute_similar_images(
        image_tensor, num_images=5, embedding=embedding, device=device
    )
    # Return JSON format response
    return (
        json.dumps({"indices_list": indices_list[0]}),
        200,
        {"ContentType": "application/json"},
    )


# Main program entry point
if __name__ == "__main__":
    # Start Flask application, disable debug mode, listen on port 9000
    app.run(debug=False, port=9000)