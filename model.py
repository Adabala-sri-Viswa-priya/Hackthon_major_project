import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


# Function to load and preprocess the dataset
def load_data(data_dir):
    real_images, fake_images = [], []
    for category in ["training_real", "training_fake"]:
        category_path = os.path.join(data_dir, category)
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(category_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    # Resize image to a fixed size (e.g., 224x224)
                    image = cv2.resize(image, (224, 224))
                    # Normalize pixel values to range [0, 1]
                    image = image.astype('float32') / 255.0
                    if category == "training_real":
                        real_images.append(image)
                    else:
                        fake_images.append(image)
    return np.array(real_images), np.array(fake_images)

# Load the dataset
real_images, fake_images = load_data("aiproject")

# Check if both categories have images
if len(real_images) > 0 and len(fake_images) > 0:
    # Create labels for real and fake images
    real_labels = np.ones(len(real_images))
    fake_labels = np.zeros(len(fake_images))

    # Split data into training and testing sets
    real_train, real_test, real_train_labels, real_test_labels = train_test_split(real_images, real_labels, test_size=0.2, random_state=42)
    fake_train, fake_test, fake_train_labels, fake_test_labels = train_test_split(fake_images, fake_labels, test_size=0.2, random_state=42)

    # Concatenate real and fake training data and labels
    X_train = np.concatenate((real_train, fake_train), axis=0)
    y_train = np.concatenate((real_train_labels, fake_train_labels), axis=0)

    # Concatenate real and fake testing data and labels
    X_test = np.concatenate((real_test, fake_test), axis=0)
    y_test = np.concatenate((real_test_labels, fake_test_labels), axis=0)

    # Flatten the image data for traditional machine learning algorithms
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Define and train the classifier
    classifier = SVC(kernel='linear', random_state=42)
    classifier.fit(X_train_flat, y_train)

    # Evaluate the classifier
    y_pred = classifier.predict(X_test_flat)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)

    # Save the trained classifier
    joblib.dump(classifier, "fake_image_detector_model.pkl")

    # Load the saved classifier
    classifier = joblib.load("fake_image_detector_model.pkl")

    # Function to predict whether an image is real or fake
    def predict_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to load the image.")
            return None

        image = cv2.resize(image, (224, 224))  # Resize image to match model input shape
        image = image.astype('float32') / 255.0  # Normalize pixel values
        image_flat = image.reshape(1, -1)  # Flatten the image data
        prediction = classifier.predict(image_flat)[0]
        return "Real" if prediction == 1 else "Fake"

    # Example usage: predict whether an image is real or fake
    image_path = "static\\test2pic.jpeg"

    prediction = predict_image(image_path)
    print("Prediction:", prediction)