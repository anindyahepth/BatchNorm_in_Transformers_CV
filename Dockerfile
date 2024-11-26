FROM python:3.9

WORKDIR /usr/src/app

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Let' only copy the required files and folders
ADD ./model ./model
COPY ./app.py ./app.py
ADD ./templates ./templates
ADD ./static ./static
ADD ./model_pth_dir ./model_pth_dir

EXPOSE 5000

CMD ["python", "app.py" ]
