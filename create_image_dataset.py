import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files("prondeau/the-car-connection-picture-dataset", path='dataset_images/all/', unzip=True)