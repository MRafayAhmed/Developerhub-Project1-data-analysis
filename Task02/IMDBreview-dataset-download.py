import kagglehub

# Download latest version
path = kagglehub.dataset_download("pawankumargunjan/imdb-review")

print("Path to dataset files:", path)