# OC-P9
# Recommendation System

## Project scenario

My Content is a fictitious start-up that wants to encourage reading by recommending relevant content for its users.

**Objective** : Develop a Minimum Viable Product (**MVP**) of a recommendation system for My Content, allowing relevant articles and books to be recommended to users.

Solutions to explore:
- Test **different** modeling approaches
    - **Collaborative Filtering** - a recommendation technique that uses user interactions with items to make suggestions
    - **Content-Based** - which analyzes the characteristics of the items themselves to make suggestions
- **Data** at our disposal:
    - Use of open-source data from an information portal containing user interactions with articles, as well as information about the articles themselves
    - https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom
- **Recommendation System**:
    - In this upstream phase of the project, develop a simple tool
    - For a given user: recommend 5 articles
- **Application**:
    - Simple user interface
    - Input: list of users, more precisely user_ids
    - Output: list of 5 recommended article_ids
- MVP **Architecture**:
    - Opportunity to implement a serverless solution, with Azure Functions
    - 2 proposals:
        - Solution 1:
            - Backend: classic API deployed on servers
            - Frontend: classic UI deployed on servers
            - Link between the two: function app that calls the API and returns the result
        - Solution 2:
            - Backend: function app
            - Frontend: classic UI deployed on servers
    - **Here: we will push the logic to the maximum by also managing the UI using Azure Functions**
- **Target** Architecture:
    - Think about the next steps: What about adding new users and new items?
    - What impact on architectural solutions?


## Run the app

### Local

- Go in the application directory
```bash
cd "Patin_Clement_1_application_052024"
```
- Launch Azurite
- Make sure to upload to Azurite the necessary files (those in `Patin_Clement_2_scripts_052024/mySaves/prod_files`) in a container named "ocp9container"
- Lauch app
```bash
func start
```

### to Azure Function App
For *Azure* deployment, a .py script is in  `Patin_Clement_2_scripts_052024`
- Go in the scripts directory 
```bash
cd "Patin_Clement_2_scripts_052024"
```
- Edit script `c_deploy_function_app.py` with your own '<your_subscription_id>' and '<your_assignee_object_id>'
- Launch script
```bash
python c_deploy_function_app.py
```