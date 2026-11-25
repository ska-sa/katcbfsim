#!/bin/bash
# Compile a downstream package requirements .in file to a .txt file.
# The output .txt file is written alongside the .in file with the
# same base name but .txt extension.
#
# Run this whenever any .in file changes, then commit the .txt file.
#
# Usage:
#   # CPU-only package (base constraints only):
#   ./compile-downstream-requirements.sh <requirements.in> \
#       -c <base-requirements.txt>
#
#   # GPU package (base + gpu constraints):
#   ./compile-downstream-requirements.sh <requirements.in> \
#       -c <base-requirements.txt> \
#       -c <gpu-requirements.txt>

set -e

# Python version used for compiling requirements.
# Change this if the base image Python version changes.
PYTHON_VERSION="3.10"

if [ -z "$1" ]; then
    echo "Usage: $0 <requirements.in> -c <base-requirements.txt> [-c <gpu-requirements.txt>]"
    exit 1
fi

IN_FILE="$(realpath "$1")"
shift

if [ ! -f "${IN_FILE}" ]; then
    echo "Error: ${IN_FILE} not found."
    exit 1
fi

IN_DIR="$(dirname "${IN_FILE}")"
IN_NAME="$(basename "${IN_FILE}")"
OUT_NAME="${IN_NAME%.in}.txt"

# Collect constraint arguments
CONSTRAINTS=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -c)
            shift
            CONSTRAINTS="${CONSTRAINTS} --constraint $(realpath "$1")"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 <requirements.in> -c <base-requirements.txt> [-c <gpu-requirements.txt>]"
            exit 1
            ;;
    esac
done

if [ -z "${CONSTRAINTS}" ]; then
    echo "Error: at least one constraint file (-c) is required."
    exit 1
fi

echo "Compiling ${IN_NAME} -> ${OUT_NAME}..."
(cd "${IN_DIR}" && uv pip compile \
    --python-version "${PYTHON_VERSION}" \
    --upgrade \
    ${CONSTRAINTS} \
    "${IN_NAME}" \
    -o "${OUT_NAME}")

# Strip full paths from the output file leaving only filenames
# for both the header comment and the via comments
sed -i 's|--constraint [^ ]*/docker-base-build/|--constraint docker-base-build/|g' "${IN_DIR}/${OUT_NAME}"
sed -i 's|--constraint [^ ]*/docker-base-gpu-build/|--constraint docker-base-gpu-build/|g' "${IN_DIR}/${OUT_NAME}"
sed -i 's|-c [^ ]*/docker-base-build/|-c docker-base-build/|g' "${IN_DIR}/${OUT_NAME}"
sed -i 's|-c [^ ]*/docker-base-gpu-build/|-c docker-base-gpu-build/|g' "${IN_DIR}/${OUT_NAME}"

echo "Done: ${IN_DIR}/${OUT_NAME}"
echo "Commit ${OUT_NAME} to the repository."
