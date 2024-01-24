# Customer Satisfaction Model Deployment using MLOPS

![Screenshot 2024-01-24 at 1 44 00 PM](https://github.com/anthonymoub/MLOPS-Deployment-Pipeline/assets/112438562/2e4af17e-842e-46a6-a073-9332f491ddea)

**Problem Statement**: For a given customer's historical data, the task is to predict the review score for the next order or purchase. The dataset used is the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). This dataset has information on 100,000 orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing charges from various dimensions: from order status, price, payment, freight performance to customer location, product attributes and finally, reviews written by customers. The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. In order to achieve this in a real-world scenario, [ZenML](https://zenml.io/) was used to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.

**Main Objective** 

The main objective of this project is to automate the end-to-end process of model training, evaluation, and deployment using a continuous deployment pipeline. At its core, the project integrates with MLflow, a popular open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment.
The purpose of this project is to demonstrate how tools like [ZenML](https://github.com/zenml-io/zenml) and [MLflow](https://mlflow.org/) could be used to build and deploy machine learning pipelines.

**WorkFlow**

![Screenshot 2024-01-24 at 1 52 20 PM](https://github.com/anthonymoub/MLOPS-Deployment-Pipeline/assets/112438562/cbb0cbee-4956-422e-855d-457d9f0505c6)

The diagram above illustrates the workflow of the whole pipeline. It begins with the "Data Ingester" phase, where data is sourced and ingested into the system. Next, in the "Data Clean" phase, the ingested data is processed and cleaned, ensuring it is in the correct format for model training. The "Trainer" phase involves the application of machine learning algorithms to train models on the cleaned data. Following training, the "Evaluate" phase assesses the model's performance using various metrics. Based on evaluation results, "Deployment Triggers" determine whether the model meets the criteria for deployment. If so, the model progresses to the "(Re) Deployment" phase, where it is deployed for inference. The deployed model is hosted as a "Model Service" in the cloud, allowing for scalability and accessibility. Finally, the "Inference using Streamlit" phase enables users to interact with the model through a Streamlit application, providing a user-friendly interface for sending input data to the model and receiving predictions.

**Streamlit App**

The deployment pipeline has been implemented into a Streamlit app, where you can to observe its functionality in the video belo:

[![Alt text for image](https://github.com/anthonymoub/MLOPS-Deployment-Pipeline/assets/112438562/e859ae46-72e1-4dcc-adb1-0d521fed4a71)](https://www.youtube.com/watch?v=ycbsLh6IHOQ "Title of video")



