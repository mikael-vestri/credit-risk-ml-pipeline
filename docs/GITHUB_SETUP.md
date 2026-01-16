# GitHub Repository Setup Guide

## Check Current Status

To verify if the repository is already connected to GitHub:

```bash
# View configured remote repositories
git remote -v

# If there's no output, no remote is configured
```

## Step 1: Create Repository on GitHub

1. Go to [GitHub](https://github.com) and log in
2. Click **"New repository"** (or go to https://github.com/new)
3. Configure:
   - **Repository name**: `credit-risk-ml-pipeline` (or another name of your choice)
   - **Description**: "Production-ready ML pipeline for predicting loan default probability"
   - **Visibility**: Private (recommended) or Public
   - **DO NOT check** "Initialize with README" (we already have one)
   - **DO NOT add** .gitignore or license (we already have them)
4. Click **"Create repository"**

## Step 2: Connect Local Repository to GitHub

After creating the repository on GitHub, you'll see instructions. Use one of these options:

### Option A: HTTPS (Recommended for beginners)

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-ml-pipeline.git

# Verify
git remote -v
```

### Option B: SSH (If you've already configured SSH keys)

```bash
# Add remote
git remote add origin git@github.com:YOUR_USERNAME/credit-risk-ml-pipeline.git

# Verify
git remote -v
```

## Step 3: Make Initial Commit

```bash
# Add all files
git add .

# Make initial commit
git commit -m "feat: initial commit - ML pipeline through Step 10 (Model Interpretability)"

# Rename branch master to main (if necessary)
git branch -M main

# Initial push
git push -u origin main
```

## Step 4: Verify Connection

```bash
# Verify remote
git remote -v

# Should show something like:
# origin  https://github.com/YOUR_USERNAME/credit-risk-ml-pipeline.git (fetch)
# origin  https://github.com/YOUR_USERNAME/credit-risk-ml-pipeline.git (push)

# Check status
git status
```

## Step 5: Create Branching Strategy

```bash
# Create develop branch
git checkout -b develop
git push -u origin develop

# Create branch for Step 11
git checkout -b feature/step-11-production-deployment
git push -u origin feature/step-11-production-deployment

# Return to main
git checkout main
```

## Recommended Branch Structure

```
main (production)
  └── develop (development)
      ├── feature/step-11-production-deployment
      ├── feature/step-12-monitoring
      └── feature/bugfix-*
```

## Useful Commands

### Verify if connected to the correct remote
```bash
git remote -v
git remote get-url origin
```

### Change remote URL (if necessary)
```bash
git remote set-url origin https://github.com/NEW_USERNAME/NEW_REPO.git
```

### Remove remote (if necessary)
```bash
git remote remove origin
```

### View local and remote branches
```bash
git branch -a
```
