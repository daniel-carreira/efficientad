ARG CUDA_VERSION=12.4
FROM nvidia/cuda:${CUDA_VERSION}.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# ESSENCIALS
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    cmake \
    curl \
    git \
    tzdata \
    x11-utils \
    libgtk2.0-dev \
    libgl1-mesa-glx

# > TZDATA
RUN ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# > VENV
RUN python3.11 -m venv /opt/efficientad-venv
ENV PATH="/opt/efficientad-venv/bin:$PATH"
RUN ln -sf /opt/efficientad-venv/bin/python /usr/bin/python && \
    ln -sf /opt/efficientad-venv/bin/python /usr/bin/python3

# > GSTREAMER
RUN apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio

# > PIP
COPY requirements.txt .
RUN pip install -r requirements.txt --use-pep517

# > OPENCV
RUN mkdir -p opencv && cd opencv && \
    git clone --branch 4.11.0 https://github.com/opencv/opencv.git && \
    git clone --branch 4.11.0 https://github.com/opencv/opencv_contrib.git && \
    cd opencv && mkdir -p build && cd build && \
    cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
        -D PYTHON_EXECUTABLE=$(which python) \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D BUILD_opencv_python3=ON \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D WITH_CUDA=ON \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D WITH_CUBLAS=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_FFMPEG=ON \
        -D OPENCV_DNN_CUDA=OFF \
        -D BUILD_opencv_apps=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D PKG_CONFIG_ARGN="--dont-define-prefix" \
        -D Python3_FIND_STRATEGY=LOCATION \
        .. && \
        make -j"$(nproc)" && \
        make install && \
        ldconfig

# CLEANUP
RUN rm -rf *
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ENTRYPOINT [ "tail", "-f", "/dev/null" ]