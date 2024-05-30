from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import os

# Initialize DefaultAzureCredential
credential = DefaultAzureCredential()

# Replace with your storage account name
account_name = "ocp9storage"
blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=credential)

# Create a container
container_name = "ocp9container"
if container_name not in [container["name"] for container in blob_service_client.list_containers()] :
    blob_service_client.create_container(container_name)

# get files list
dir_path = "mySaves/prod_files"
blob_names_list = os.listdir(dir_path)

print(blob_names_list)

# Upload files
for blob_name in blob_names_list :
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(os.path.join(dir_path, blob_name), "rb") as f:
        blob_client.upload_blob(f)
    print(f"{blob_name} uploaded to container {container_name}")
