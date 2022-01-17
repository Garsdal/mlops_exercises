from google.cloud import storage
# Upload function
def upload_blob(bucket_name, source_file_name, destination_blob_name):
            """Uploads a file to the bucket."""
            # The ID of your GCS bucket
            # bucket_name = "your-bucket-name"
            # The path to your file to upload
            # source_file_name = "local/path/to/file"
            # The ID of your GCS object
            # destination_blob_name = "storage-object-name"

            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)

            blob.upload_from_filename(source_file_name)

            print(
                "Bucket upload: File {} uploaded to {}.".format(
                    source_file_name, destination_blob_name
                )
            )

# Load data
import numpy as np
from sklearn import datasets

iris_X, iris_y = datasets.load_iris(return_X_y=True)

# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
knn.predict(iris_X_test)

print(iris_X_test)
print(knn.predict(iris_X_test))

# save model
# import pickle
# with open('model.pkl', 'wb') as file:
#     pickle.dump(knn, file)

UPLOAD = False
if UPLOAD: 
    bucket_name = "sklearn-cloud-function"
    source_file_name = "model.pkl"
    destination_blob_name = "model.pkl"
    upload_blob(bucket_name, source_file_name, destination_blob_name)

