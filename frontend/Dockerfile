# Build stage
FROM node:16-alpine as build

# Set the working directory in the container
WORKDIR /app

# Copy package.json
COPY package.json ./

# Install dependencies
RUN yarn install --frozen-lockfile

# Copy the rest of the application code
COPY . .

# Build the app
RUN yarn build

# Production stage
FROM node:14.17.0-alpine

WORKDIR /app

# Copy built assets from the build stage
COPY --from=build /app/build ./build

# Install serve
RUN yarn global add serve

# Set the command to start the app
CMD ["serve", "-s", "build", "-l", "3000"]

# Expose port 3000
EXPOSE 3000
