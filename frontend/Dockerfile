                                                                                                                   # Use a single stage for development

FROM node:16-alpine

# Set the working directory in the container
WORKDIR /app

# Copy package.json
COPY package.json ./

# Install dependencies
RUN yarn install --frozen-lockfile

# Copy only necessary files
COPY . .

# Install serve globally
RUN npm install -g serve

# Set the command to start the app
CMD ["serve", "-s", ".", "-l", "3000"]

# Expose port 3000
EXPOSE 3000
