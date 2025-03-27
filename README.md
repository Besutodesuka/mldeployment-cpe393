# about Repo
## Project description
this project is part of CPE393 ml operation at KMUTT. this is about deploying the model using docker container which using flask to deploy API for inferencing. this project have structure as follow

```json
-- use case 1
    |
    |--app
    |      |--model
    |      |--app.py (API)
    |      |--requirements.txt
    |--docker-compose.yaml
    |--train.py
-- use case 2
    | ...

this repository offer you 2 example with california house pricing and iris dataset
```

## Setup steps
1. first you need to go to your desired folder for example:
```shell
cd iris
```
2. install required package
```shell
pip install ./app/requirements.txt
```
3. train some model
```shell
python train.py
```
4. deploy
```shell
docker compose up
```

and you good tto go service is at localhost:9000
## Sample API request and response
here is some request list
1. GET /health
this is used to check if api is still operating

2. POST /predict 
this api is used to call AI prediction and pay load should be
```json
header: "Content-Type" : "application/json"
body: {
    "features": 
}
```
you can check format of each usecase on test.txt file on each folder
# mldeployment-cpe393 (instruction)

## model export
Run train.py. (model.pkl will be saved in app folder)

## Go to the directory in terminal
cd "project folder directory"

## Build Docker image
docker build -t ml-model .

## Run Docker container
docker run -p 9000:9000 ml-model

## Test the API in new terminal

curl -X POST http://localhost:9000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

expected output

{"prediction": 0}





