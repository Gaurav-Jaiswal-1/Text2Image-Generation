To push your project to GitHub, follow these steps:

### 1. **Initialize Git Repository**
Navigate to your project directory and initialize a Git repository:

```bash
cd /path/to/your/project
git init
```

### 2. **Add All Project Files**
Add all the files to the Git staging area:

```bash
git add .
```

### 3. **Commit the Changes**
Create a commit with a message describing your changes:

```bash
git commit -m "Initial commit"
```

### 4. **Create a GitHub Repository**
Go to GitHub, create a new repository, and don't initialize it with a README file (since you've already committed your files locally).

### 5. **Add GitHub Remote**
After creating the repository, link it to your local project by adding a remote origin. Use the URL of the GitHub repository (replace `<username>` and `<repository>`):

```bash
git remote add origin https://github.com/<username>/<repository>.git
```

### 6. **Push to GitHub**
Push the project to GitHub:

```bash
git push -u origin master
```

Your project should now be live on GitHub!