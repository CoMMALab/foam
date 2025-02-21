# foam Docker Testing Configuration
A custom dockerfile designed to containerize foam on Ubuntu 24.04 


# Features 
* automatically installs and builds foam from source
* Creates an organized environment ready for foam testing on a clean system with dependency management
* Can be directly developed on via VSCode

### Step 1: Install Docker Desktop and run Docker Desktop. 
Docker Desktop comes with all components of Docker that are necessary to make our docker containers, as well as a nice UI to help us manage our containers and images. 
https://www.docker.com/products/docker-desktop/
### Step 2: Download the docker file from this repository. 
Download `Dockerfile` and add it to a folder
### Step 3: Retrieve a Personal Access Token (PAT)
As foam is currently a private repository, a PAT is required to clone the repo within a docker environment
Steps:
* Navigate to your Github profile
* Click settings
* Click developer tools
* Click Personal Access Tokens
* Click Tokens (classic)
* Generate a new classic token. Give it repo access, and modify expiration settings if necessary
* Copy the generated PAT to a file named PAT.txt within the same directory as the Dockerfile

### Step 4: Building the Docker image
First set the DOCKER_BUILDKIT environment variable. This speeds up Docker builds and allows utilizing secrets for private repository access.
On windows, run
```
$env:DOCKER_BUILDKIT = "1"
```

On Linux and macOS, run
```
export DOCKER_BUILDKIT=1
```

To build the image, run:
```docker build --secret id=github_token,src=PAT.txt -t foam-image .```

### Step 5: Running the docker container
To run the docker container from the image, run
```docker run -it -v "$(pwd)/:/foam"  --name=foam foam-image```

To re-enter the docker container, run
```docker exec -it foam /bin/bash```

To start and stop a container, run
```docker start foam```
```docker stop foam```

### Step 6: Compile Libraries
Run the following to compile submodules:
```
cd foam
cmake -Bbuild -GNinja .
cmake --build build/
```

For more details on display forwarding and other docker management commands, see the following:
https://github.com/saiccoumar/PX4_Docker_Config
