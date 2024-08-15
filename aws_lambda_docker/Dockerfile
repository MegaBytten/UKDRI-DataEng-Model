# use AWS-Lambda python runtime image
FROM public.ecr.aws/lambda/python:3.12.2024.08.09.13

# Copy Files: requirements.txt, model.pkl, code (lambda_function.py)
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY tenancy_logreg_model.pkl ${LAMBDA_TASK_ROOT}
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt


# Set the CMD to your handler 
CMD [ "lambda_function.handler" ]