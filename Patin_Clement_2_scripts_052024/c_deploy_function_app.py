# pip install azure-identity azure-mgmt-resource azure-mgmt-storage azure-mgmt-web

#
import os
import re
import subprocess
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobServiceClient

# subscription_id = '<your_subscription_id>'
subscription_id = '9960f4c5-c6ef-4c0a-91a7-542efbbe47da'
resource_group_name = 'ocp9'
location = 'westeurope'
storage_account_name = 'ocp9storage'
container_name = "ocp9container"
app_service_plan_name = 'ocp9plan'
sku = "EP1"
function_app_name = 'ocp9app'
# user_object_id = '<your_assignee_object_id>'
user_object_id = 'c69cbee2-5b4e-4d76-87f0-b5e7c9ccd24b'


class recommender_deployer :
    """
    Manages Azure resources including resource groups, storage accounts, app service plans, 
    function apps, and role assignments.
    """
    
    def __init__(self, subscription_id, user_object_id):
        """
        Initialize the recommender_deployer with the necessary credentials and clients.

        :param subscription_id: Azure subscription ID.
        :param assignee_object_id: Object ID of the assignee.
        :param role_id: Role ID for role assignment.
        """
        # acquire a credential object
        self.credential = DefaultAzureCredential()
        # initialize attributes
        self.subscription_id = subscription_id
        self.user_object_id = user_object_id
        # initialize clients
        self.resource_client = ResourceManagementClient(self.credential, subscription_id)
        self.storage_client = StorageManagementClient(self.credential, subscription_id)
        # self.authorization_client = AuthorizationManagementClient(self.credential, subscription_id)
        # self.web_client = WebSiteManagementClient(self.credential, subscription_id)

    def create_resource_group(self, resource_group_name, location):
        """
        Create a resource group.

        parameters :
        ------------
        - resource_group_name - string : Name of the resource group.
        - location - string : Azure region (e.g., 'westeurope').
        """
        # create resource group
        self.resource_group_name = resource_group_name
        self.resource_client.resource_groups.create_or_update(self.resource_group_name, {'location': location})
        print(f'----> Resource group {self.resource_group_name} created')

    def create_storage_account(self, storage_account_name, location):
        """
        Create a storage account.

        parameters :
        ------------
        - resource_group_name - string : Name of the resource group.
        - storage_account_name - string : Name of the storage account.
        - param location - Azure region (e.g., 'westeurope').
        """
        # Check if the account name is available. 
        availability_result = self.storage_client.storage_accounts.check_name_availability(
            { "name": storage_account_name }
        )
        if not availability_result.name_available:
            print(f"----> Storage name {storage_account_name} is already in use. Try another name.")
            exit()

        # create storage acount
        self.storage_account_name = storage_account_name

        poller = self.storage_client.storage_accounts.begin_create(
            self.resource_group_name,
            storage_account_name,
            {
                'sku': {'name': 'Standard_LRS'},
                'kind': 'StorageV2',
                'location': location
            }
        )
        # Long-running operations return a poller object; calling poller.result() waits for completion.
        account_result = poller.result()
        print(f"----> Provisioned storage account {account_result.name}")

    def assign_role_to_user(self):
        """
        Assign a role to a principal.
        """
        # assign role to user
        subprocess.call([
            "az", "role", "assignment", "create", 
            "--assignee-object-id", self.user_object_id, 
            "--assignee-principal-type", "User",
            "--scope", f'/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group_name}',
            "--role", "Storage Blob Data Contributor"
            ], shell=True)

        # role_assignment_params = {
        #     'role_definition_id': f'/subscriptions/{self.subscription_id}/providers/Microsoft.Authorization/roleDefinitions/{self.role_id}',
        #     'principal_id': principal_id,
        #     'principal_type': principal_type,
        #     'scope': resourceGroups/
        # }
        # parameters = self.authorization_client.role_assignments.models.RoleAssignmentCreateParameters(

        # )
        # return self.authorization_client.role_assignments.create(
        #     scope=f'/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group_name}',
        #     role_assignment_name=str(uuid.uuid4()),
        #     parameters, 

        # )
        # return self.authorization_client.role_assignments(
        #     scope=role_assignment_params['scope'],
        #     role_assignment_name=str(uuid.uuid4()),
        #     parameters=role_assignment_params
        # )
    
        

    def load_files(self, container_name):
        """
        xxxxxxxxxxxxxxx
        """
        # initialize blob client
        blob_service_client = BlobServiceClient(account_url=f"https://{self.storage_account_name}.blob.core.windows.net", credential=self.credential)
        # Create a container
        self.container_name = container_name
        if container_name not in [container["name"] for container in blob_service_client.list_containers()] :
            blob_service_client.create_container(container_name)

        # get files list
        dir_path = "mySaves/prod_files"
        blob_names_list = os.listdir(dir_path)

        # Upload files
        for blob_name in blob_names_list :
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            with open(os.path.join(dir_path, blob_name), "rb") as f:
                blob_client.upload_blob(f)
            print(f"----> {blob_name} uploaded to container {container_name}")



    def create_app_service_plan(self, app_service_plan_name, sku, location):
        """
        Create an app service plan.

        parameters :
        ------------
        - app_service_plan_name - string
        - sku - string : can be an Elastice Premium Plan (eg "EP1") or a Dedicated App Service Plan (eg "B2") (in that case, can be shared with another app, like a webapp)
        - location - string : Azure region (e.g., 'westeurope')
        """
        # create a app service plan
        self.app_service_plan_name = app_service_plan_name

        subprocess.call([
            "az", "functionapp", "plan", "create", 
            "--name", self.app_service_plan_name, 
            "--resource-group", self.resource_group_name, 
            "--location", location, 
            "--is-linux", 
            "--sku", sku
            ], shell=True)
        
        print(f"----> Function app service plan {app_service_plan_name} created, with sku {sku}")
        

    def create_function_app(self, function_app_name):
        """
        Create a function app.

        Parameter :
        -----------
        - function_app_name -string : Name of the function app
        """
        # create a function app
        self.function_app_name = function_app_name

        subprocess.call([
            "az", "functionapp", "create", 
            "--name", self.function_app_name, 
            "--resource-group", self.resource_group_name, 
            "--os-type", "Linux", 
            "--runtime", "python", 
            "--runtime-version", "3.11", 
            "--functions-version", "4", 
            "--storage-account", self.storage_account_name, 
            "--plan", self.app_service_plan_name 
            ], shell=True)
        
        print(f"----> Function app {function_app_name} created")

    def assign_identity_to_function_app(self):
        """
        Assign a managed identity to the function app.
        """
        # enable managed identity
        output = subprocess.check_output([
            "az", "functionapp", "identity", "assign", 
            "--resource-group", self.resource_group_name, 
            "--name", self.function_app_name
            ], shell=True)
        # decode output
        output = output.decode("utf-8")
        # extract principalId
        pattern = r'(?<="principalId":\s)(?:")(\S+)(?:")'
        principalId = re.findall(pattern=pattern, string=output)[0]
        
        print("----> App principal id :", principalId)

        # assign role to managed identity
        subprocess.call([
            "az", "role", "assignment", "create", 
            "--assignee", principalId, 
            "--scope", f'/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group_name}',
            "--role", "Storage Blob Data Contributor"
            ], shell=True)
        
        print(f"----> Managed identity for {self.function_app_name} created, with role 'Storage Blob Data Contributor'")

    def set_function_app_connection_setting(self) :
        """
        Set application settings for the function app, for connection purpose.
        """
        # create a app setting for identity-based connection
        subprocess.call([
            "az", "functionapp", "config", "appsettings", "set", 
            "--name", self.function_app_name,
            "--resource-group", self.resource_group_name,
            "--settings", f"MY_CONNECTION_SETTING__serviceUri=https://{self.storage_account_name}.blob.core.windows.net"
            ], shell=True)
        
        print(f"----> Connection setting for function app {self.function_app_name} created")
        

    def publish(self) :
        """
        publish function app
        """
        # change working directory
        olddir = os.getcwd()
        parent_path = os.path.abspath("..")
        app_source_code_path = (os.path.join(parent_path,"Patin_Clement_1_application_052024"))
        os.chdir(app_source_code_path)

        # publish function app
        subprocess.call([
             "func", "azure", "functionapp", "publish", self.function_app_name
             ], shell=True)
        
        os.chdir(olddir)
        
        print(f'----> Function app {self.function_app_name} is published')


    def kill_resource(self) :
        """
        delete resource group
        """
        # delete all resources (storage, plan, function app)
        # subprocess.call([
        #      "az", "group", "delete", "--name", self.resource_group_name
        #      ], shell=True)
        
        poller = self.resource_client.resource_groups.begin_delete(self.resource_group_name)
        
        print(poller.result())
        print(f'----> Resource group {self.resource_group_name} is deleted')
        print(f'----> (along with {self.storage_account_name}, {self.app_service_plan_name} and {self.function_app_name})')


def main():
    """
    Main function to execute the Azure resource creation and configuration process.
    """
    # initiate a recommender_deployer
    recDeployer = recommender_deployer(
        subscription_id=subscription_id,
        user_object_id=user_object_id,
        )
    # Create Resource Group
    recDeployer.create_resource_group(
        resource_group_name=resource_group_name,
        location=location
        )
    # Create Storage Account
    recDeployer.create_storage_account(
        storage_account_name=storage_account_name,
        location=location
        )
    # Assign Role to User
    recDeployer.assign_role_to_user()
    # Load files to blob storage
    recDeployer.load_files(
        container_name=container_name
        )
    # Create App Service Plan
    recDeployer.create_app_service_plan(
        app_service_plan_name=app_service_plan_name, 
        sku=sku,
        location=location
        )
    # Create Function App
    recDeployer.create_function_app(
        function_app_name=function_app_name
        )
    # Assign Managed Identity to Function App
    recDeployer.assign_identity_to_function_app()
    # Create a function app setting for connection
    recDeployer.set_function_app_connection_setting()
    # publish webapp
    recDeployer.publish()

if __name__ == "__main__":
    main()
