FROM public.ecr.aws/lambda/python:3.11

# Install system build dependencies first (cached unless this changes)
RUN yum install -y gcc10 gcc10-c++ python3-devel && \
    ln -sf /usr/bin/gcc10 /usr/bin/gcc && \
    ln -sf /usr/bin/g++10 /usr/bin/g++

# Install Python dependencies before copying source code.
# This layer is cached as long as requirements.txt doesn't change,
# so source-only edits won't trigger a full pip re-install.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code last so the expensive dependency layer stays cached.
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY lambda_app.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_app.lambda_handler"]
