# This is a course project for CS291I: Introduction to Robotics Vision.

## Project Overview



## Setup
Create a conda environment (or virtualenv):
```
conda create -n smartllm python==3.9
```

Install dependencies:
```
pip install -r requirments.txt
```

## Creating LLM API Key
### for OpenAI GPT
The code relies on OpenAI API. Create an API Key at https://platform.openai.com/.

Create a file named ```api_key.txt``` in the root folder of the project and paste your OpenAI Key in the file. 

### for DeepSeek
The code relies on DeepSeek API. Create an API Key at https://deepseek.com/apikey.

Create a file named ```deepseek_api_key.txt``` in the root folder of the project and paste your DeepSeek Key in the file. 

### for Gemini
The code relies on Gemini API. Create an API Key at https://ai.google.dev/gemini-api/docs/quickstart.

Create a file named ```gemini_api_key.txt``` in the root folder of the project and paste your Gemini Key in the file. 

### for Llama
The code relies on Llama API. Create an API Key at https://llama.com/api.

Create a file named ```llama_api_key.txt``` in the root folder of the project and paste your Llama Key in the file. 


## Running Script
Run the following command to generate output execuate python scripts to perform the tasks in the given AI2Thor floor plans. 

Refer to https://ai2thor.allenai.org/demo for the layout of various AI2Thor floor plans.
```
python3 scripts/{run_llm.py} --floor-plan {floor_plan_no}
```
Note: Refer to the script for running it on different versions of GPT models and changing the test dataset. 

The above script should generate the executable code and store it in the ```logs``` folder.


Run the following script to execute the above generated scripts and execute it in an AI2THOR environment. 

The script requires command which needs to be executed as parameter. ```command``` needs to be the folder name in the ```logs``` folder where the executable plans generated are stored. 
```
python3 scripts/execute_plan.py --command {command}
```

## Dataset
The repository contains numerous commands and robots with various skill sets to perform heterogenous robot tasks. 

Refer to ```data\final_test\``` for the various tasks, robots available for the tasks, and the final state of the environment after the task for evaluation. 

The file name corresponds to the AI2THOR floor plans where the task will be executed. 

Refer to ```resources\robots.py``` for the list of robots used in the final test and the skills possessed by each robot. 



```
