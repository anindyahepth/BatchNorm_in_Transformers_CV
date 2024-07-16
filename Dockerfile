FROM python:3.9

WORKDIR /anindyahepth/BatchNorm_in_Transformers_CV/

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Let' only copy the required files and folders
ADD ./model
COPY ./app.py ./
ADD ./templates
ADD ./static

EXPOSE 5000

CMD ["python", "app.py" ]
