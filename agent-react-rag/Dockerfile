FROM public.ecr.aws/lambda/python:3.11

# Install runtime dependencies
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel
RUN yum install -y gcc10 gcc10-c++ python3-devel && \
    ln -sf /usr/bin/gcc10 /usr/bin/gcc && \
    ln -sf /usr/bin/g++10 /usr/bin/g++ && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ${LAMBDA_TASK_ROOT}

# Lambda handler
CMD ["lambda_app.lambda_handler"]
