# Use a base image
FROM node:18

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN npm install --save-dev @types/react @types/react-dom
RUN npm install -D tailwindcss postcss autoprefixer
RUN npx tailwindcss init -p
RUN npm install

# Expose port
EXPOSE 8000

# Run app
CMD ["node", "server.js"]
