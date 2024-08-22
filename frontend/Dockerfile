# Build stage
FROM node:16-alpine as build

# Set the working directory in the container
WORKDIR /app

# Copy package.json
COPY package.json ./

# Install dependencies
RUN yarn install --frozen-lockfile

# Copy only necessary files
COPY . .

# Build the app
RUN yarn build

# Production stage
FROM node:16-alpine

WORKDIR /app

# Copy built assets from the build stage
COPY --from=build /app/build ./build

# Install serve globally
RUN npm install -g serve

# Set the command to start the app
CMD ["serve", "-s", "build", "-l", "3000"]

# Expose port 3000
EXPOSE 3000
