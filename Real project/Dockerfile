# Use Python 3.12 base image
FROM python:3.12

# Create a user and set environment variables
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory inside the container
WORKDIR $HOME/app

# Copy the project files (including pyproject.toml)
COPY --chown=user . .

# Install dependencies
RUN pip install --no-cache-dir .

# Set the default command to run Chainlit
CMD ["chainlit", "run", "main.py", "--port", "7860", "--host", "0.0.0.0"]

