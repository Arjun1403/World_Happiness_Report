#Using official python 3.8
FROM python:3.8

#setting up working directory
WORKDIR /app

#copies package.json file to the app folder
COPY requirements.txt requirements.txt

#install the dependicies specified in the requirements.txt
RUN pip install -r requirements.txt


#Expose the port 3000 once the container has launched
EXPOSE 3000


#run specifies command within the container
CMD [ "python", "visualization_project.py" ]

#copy rest of the application source code from host to image
COPY . .

