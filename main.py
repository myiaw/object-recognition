import cv2 as cv
import numpy as np
import skimage.feature
import joblib
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import os

ratio = 0.8  # Split ratio for train and test data


def test_scenario(dt_classifier, knn_classifier, demo_images):
    dt_predictions = []
    knn_predictions = []
    actual_labels = []

    print(len(demo_images))

    for image_path in demo_images:
        image = cv.imread(image_path)
        if image is None:
            continue
        image = resize_image(image)
        dt_prediction, knn_prediction = test_model(image, dt_classifier, knn_classifier)

        dt_predictions.append(dt_prediction[0])
        knn_predictions.append(knn_prediction[0])

        actual_label = 0 if 'Cat' in image_path else 1
        # print("Actual Label: " + str(actual_label) + ", DT Prediction: " + str(dt_prediction[0]) +
        #       ", kNN Prediction: " + str(knn_prediction[0]))
        actual_labels.append(actual_label)

    print("DT Classification Report:\n",
          metrics.classification_report(actual_labels, dt_predictions, zero_division=0))
    print("kNN Classification Report:\n",
          metrics.classification_report(actual_labels, knn_predictions, zero_division=0))


def test_model(image, dt_classifier=None, knn_classifier=None):
    lbp_img = lbp_operation(image)  # Apply LBP
    hog_img = hog_operation(image, 8, 2)  # Apply H

    # Concatenate LBP and HOG vectors and reshape for prediction
    image_features = np.concatenate((lbp_img.flatten(), hog_img)).reshape(1, -1)

    dt_prediction = dt_classifier.predict(image_features)  # Predict using DT
    knn_prediction = knn_classifier.predict(image_features)  # Predict using kNN

    return dt_prediction, knn_prediction


def save_model(modelknn, modeldt, ratio=0.0):
    if ratio == 0.0:
        joblib.dump(modelknn, "knn_classifier.pkl")
        joblib.dump(modeldt, "dt_classifier.pkl")
    elif ratio == 0.8:
        joblib.dump(modeldt, "80-20dt.pkl")
        joblib.dump(modelknn, "80-20knn.pkl")
    elif ratio == 0.5:
        joblib.dump(modeldt, "50-50dt.pkl")
        joblib.dump(modelknn, "50-50knn.pkl")
    elif ratio == 0.7:
        joblib.dump(modeldt, "70-30dt.pkl")
        joblib.dump(modelknn, "70-30knn.pkl")
    pass


def load_model(ratio=0.0):
    if ratio == 0.8:
        if not os.path.exists("80-20knn.pkl"):
            print("File does not exist: " + "80-20knn.pkl")
            return None
        if not os.path.exists("80-20dt.pkl"):
            print("File does not exist: " + "80-20dt.pkl")
            return None
        return joblib.load("80-20knn.pkl"), joblib.load("80-20dt.pkl")
    elif ratio == 0.5:
        if not os.path.exists("50-50knn.pkl"):
            print("File does not exist: " + "50-50knn.pkl")
            return None
        if not os.path.exists("50-50dt.pkl"):
            print("File does not exist: " + "50-50dt.pkl")
            return None
        return joblib.load("50-50knn.pkl"), joblib.load("50-50dt.pkl")
    elif ratio == 0.7:
        if not os.path.exists("70-30knn.pkl"):
            print("File does not exist: " + "70-30knn.pkl")
            return None
        if not os.path.exists("70-30dt.pkl"):
            print("File does not exist: " + "70-30dt.pkl")
            return None
        return joblib.load("70-30knn.pkl"), joblib.load("70-30dt.pkl")


def resize_image(img, target_size=(128, 128)):
    # Resize the image to the target size
    resized_img = cv.resize(img, target_size)
    return resized_img


# Arrange the data that we got in files.
def arrange_data(ratio=0.0, amount=20000):
    cats_folder = "cats_dogs_dataset/Cat"  # Folder with cat images
    dogs_folder = "cats_dogs_dataset/Dog"  # Folder with dog images

    # Get filenames of all the images in the folders
    cats_images = os.listdir(cats_folder)
    dogs_images = os.listdir(dogs_folder)

    cats_demo = cats_images[-250:]
    dogs_demo = dogs_images[-250:]

    # Shuffle the filenames and select the specified amount for each category
    np.random.shuffle(cats_images)
    np.random.shuffle(dogs_images)
    cats_images = cats_images[:amount]
    dogs_images = dogs_images[:amount]
    print("Cats: " + str(len(cats_images)) + ", Dogs: " + str(len(dogs_images)))

    # Split the filenames into train and test based on the split ratio
    cats_train = cats_images[:int(len(cats_images) * ratio)]
    cats_test = cats_images[int(len(cats_images) * ratio):]

    dogs_train = dogs_images[:int(len(dogs_images) * ratio)]
    dogs_test = dogs_images[int(len(dogs_images) * ratio):]

    # Convert filenames to full file paths
    cats_train = [os.path.join(cats_folder, fn) for fn in cats_train]  # Add folder name to each filename
    cats_test = [os.path.join(cats_folder, fn) for fn in cats_test]  # Add folder name to each filename
    dogs_train = [os.path.join(dogs_folder, fn) for fn in dogs_train]  # Add folder name to each filename
    dogs_test = [os.path.join(dogs_folder, fn) for fn in dogs_test]  # Add folder name to each filename

    # Select the last 50 images from each category for the demo

    # Convert demo filenames to full file paths
    demo_images = [os.path.join(cats_folder, fn) for fn in cats_demo] + [os.path.join(dogs_folder, fn) for fn in
                                                                         dogs_demo]

    return cats_train, cats_test, dogs_train, dogs_test, demo_images


def lbp_operation(img):
    image = resize_image(img)  # Resize image to 128x128 px
    lbp_img = np.hstack(
        [skimage.feature.local_binary_pattern(image[:, :, i], 8, 1, method="uniform").flatten() for i in range(3)])
    return lbp_img


def hog_operation(img, cell_size, block_size):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert image to grayscale
    grayscale = resize_image(grayscale)  # Resize image to 100x100 px
    return skimage.feature.hog(grayscale, orientations=9, pixels_per_cell=(cell_size, cell_size),
                               cells_per_block=(block_size, block_size), block_norm="L2-Hys",
                               transform_sqrt=True, feature_vector=True, visualize=False)
    # Orientations: 9 bins
    # Pixels per cell: 8x8 px
    # Cells per block: 2x2 cells
    # Block norm: L2-Hys
    # Transform sqrt: Gamma correction
    # Feature vector: Flatten the final vectors
    # Visualize: Return the HOG image as well


def extract_features(cats_train, cats_test, dogs_train, dogs_test):
    # Create lists of all the images and labels
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []

    for cat in cats_train:
        print("Reading file: " + cat)
        img = cv.imread(cat)  # Read image
        if img is None:
            print("Failed to load image at: " + cat)
            continue
        lbp_img = lbp_operation(img)  # Apply LBP
        hog_img = hog_operation(img, 8, 2)  # Apply HOG
        train_images.append(np.concatenate((lbp_img.flatten(), hog_img)))  # Concatenate LBP and HOG vectors
        train_labels.append(0)  # Label 0 for cats

    for cat in cats_test:
        print("Reading file: " + cat)
        img = cv.imread(cat)  # Read image
        if img is None:
            print("Failed to load image at: " + cat)
            continue
        lbp_img = lbp_operation(img)  # Apply LBP
        hog_img = hog_operation(img, 8, 2)  # Apply HOG
        test_images.append(np.concatenate((lbp_img.flatten(), hog_img)))  # Concatenate LBP and HOG vectors
        test_labels.append(0)  # Label 0 for cats

    for dog in dogs_train:
        print("Reading file: " + dog)
        img = cv.imread(dog)  # Read image
        if img is None:
            print("Failed to load image at: " + dog)
            continue
        lbp_img = lbp_operation(img)  # Apply LBP
        hog_img = hog_operation(img, 8, 2)  # Apply HOG
        if hog_img is None:
            continue
        train_images.append(np.concatenate((lbp_img.flatten(), hog_img)))  # Concatenate LBP and HOG vectors
        train_labels.append(1)  # Label 1 for dogs

    for dog in dogs_test:
        print("Reading file: " + dog)
        img = cv.imread(dog)  # Read image
        if img is None:
            print("Failed to load image at: " + dog)
            continue
        lbp_img = lbp_operation(img)  # Apply LBP
        hog_img = hog_operation(img, 8, 2)  # Apply HOG
        test_images.append(np.concatenate((lbp_img.flatten(), hog_img)))  # Concatenate LBP and HOG vectors
        test_labels.append(1)  # Label 1 for dogs

    return train_images, train_labels, test_images, test_labels


def train_and_predict(train_images, train_labels, test_images, test_labels):
    # Train Decision Tree classifier
    dt_classifier = DecisionTreeClassifier()  # No kernel
    dt_classifier.fit(train_images, train_labels)  # Train classifier

    # Train kNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)  # 5 nearest neighbors
    knn_classifier.fit(train_images, train_labels)  # Train classifier

    # Save the trained classifiers
    save_model(knn_classifier, dt_classifier, ratio)

    # Predict labels using both classifiers
    dt_predictions = dt_classifier.predict(test_images)  # Predict using DT
    knn_predictions = knn_classifier.predict(test_images)  # Predict using kNN

    # Calculate accuracy using test labels
    dt_accuracy = metrics.accuracy_score(test_labels, dt_predictions)  # Calculate DT accuracy
    knn_accuracy = metrics.accuracy_score(test_labels, knn_predictions)  # Calculate kNN accuracy

    print("Training and prediction completed successfully.")

    return dt_predictions, knn_predictions, dt_accuracy, knn_accuracy


def main():
    models = load_model(ratio)
    cats_train, cats_test, dogs_train, dogs_test, demo_images = arrange_data(ratio, 500)
    train_images, train_labels, test_images, test_labels = extract_features(cats_train, cats_test, dogs_train,
                                                                            dogs_test)

    if models:
        knn_classifier, dt_classifier = models
        print("Models loaded successfully, proceeding to prediction...")
        dt_predictions = dt_classifier.predict(test_images)  # Predict using DT
        knn_predictions = knn_classifier.predict(test_images)  # Predict using kNN

        # Calculate accuracy using test labels
        dt_accuracy = metrics.accuracy_score(test_labels, dt_predictions)  # Calculate DT accuracy
        knn_accuracy = metrics.accuracy_score(test_labels, knn_predictions)  # Calculate kNN accuracy

        print("DT Accuracy:", dt_accuracy)
        print("kNN Accuracy:", knn_accuracy)

        test_scenario(dt_classifier, knn_classifier, demo_images)

    else:
        print("Models not found, training...")
        train_and_predict(train_images, train_labels, test_images, test_labels)

if __name__ == "__main__":
    main()