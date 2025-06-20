---
layout: single
title: "#6 Git Fundamentals for AI"
categories: Bootcamp
tag: [패스트캠퍼스, 패스트캠퍼스AI부트캠프, 업스테이지패스트캠퍼스, UpstageAILab, 국비지원, 패스트캠퍼스업스테이지에이아이랩, 패스트캠퍼스업스테이지부트캠프]
author_profile: false
---


# 🌀 Mastering Git: Essential Concepts and Commands for Every Developer

Hello again! In this post, let’s master the basics of **Git**, the most popular version control system used by developers worldwide.  
Whether you’re working solo or collaborating on large projects, Git is an essential tool for modern software development.

![Git Fundamentals](/assets/images/git_fundamentals.png)

---

## 💡 Why Git Matters?

Git is a **distributed version control system** that lets you:

- Track changes to your codebase efficiently
- Collaborate with others without conflicts
- Safely experiment with new features using branches
- Restore any previous state of your project

---

## 📚 1. What is a Git Repository?

A **repository (repo)** is a storage space for your project’s files and all their change history.

- **Local Repository**: Lives on your own computer.
- **Remote Repository**: Hosted on a server (like GitHub, GitLab) for collaboration.

---

## 🏗️ 2. Git Structure: How Does It Work?

- **Working Directory**: Where you make changes to files.
- **Staging Area (Index)**: Where you prepare changes for the next commit.
- **Repository**: Where committed versions (history) are saved permanently.

```bash
graph LR
    WD[Working Directory] --> SA[Staging Area]
    SA --> REPO[Repository]
```

---

## 🛠️ 3. Most Useful Git Commands

▶️ Creating & Cloning Repositories

```bash
git init               # Initialize a new Git repository
git clone <URL>        # Clone an existing remote repository
```

▶️ Tracking and Saving Changes

```bash
git status             # Check the current state of your repo
git add filename       # Stage a specific file
git add .              # Stage all changed files
git commit -m "msg"    # Commit staged changes with a message
```

▶️ Viewing History & Changes

```bash
git log                # View commit history
git diff               # Show file differences between commits or staging
```

▶️ Working with Branches

```bash
git branch                 # List all branches
git branch new-branch      # Create a new branch
git checkout new-branch    # Switch to a branch
git merge branch-name      # Merge another branch into current one
git branch -d branch-name  # Delete a branch
```

▶️ Working with Remote Repositories

```bash
git remote add origin <URL>  # Add a remote repository
git push origin branch-name  # Push your branch to remote
git pull origin branch-name  # Pull updates from remote
git clone <URL>              # Clone a remote repository
```
---

## 🧩 4. Basic Git Collaboration Workflow
1.	Clone the remote repository:

```bash
git clone <repo-url>
```

2.	Create and switch to a new feature branch:

```bash
git branch feature/my-feature
git checkout feature/my-feature
```

3.	Make changes, stage, and commit:

``` bash
git add .
git commit -m "Describe your changes"
```

4.	Push your branch to the remote repository:

```bash
git push origin feature/my-feature
```

5.	Open a Pull/Merge Request (on GitHub, GitLab, etc.)
- Request review, discuss changes, and merge!

---

### 📚 5. The Steps of GitHub Flow

1️⃣ Create a Branch

Start each new feature or fix by creating a branch from `main`.

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature
```
---

2️⃣ Make Commits

Work on your branch and make small, logical commits. Each commit should represent a single purpose or change.

```bash
git add .
git commit -m "Add signup form UI"
```
---

3️⃣ Open a Pull Request (PR)

When your work is ready, push your branch to GitHub and open a Pull Request (PR).
Describe what you changed and why.

```bash
git push origin feature/your-feature
```

On GitHub, click “Compare & pull request” to open a PR.

---

4️⃣ Discuss & Review Code

Collaborate with teammates—use comments for questions, suggestions, or clarifications.
Make any requested changes by committing again on your branch.
Reviewers approve or request changes before merging.

---

5️⃣ Deploy & Test

Optionally, deploy the branch to a staging/testing environment.
Automated tests (CI) and manual checks can run on PRs to catch issues early.

---

6️⃣ Merge to Main

After review and testing, merge the PR into the main branch (often using “Squash and merge” or “Rebase and merge” for a clean history).

```bash
git checkout main
git pull origin main
git merge feature/your-feature
git push origin main
```

Or simply merge from the GitHub UI.

---

7️⃣ Deploy to Production

Continuously deliver by deploying the latest main branch to production.

---

🔑 Best Practices for GitHub Flow
- Always branch from main: Keep main stable, deployable, and up to date.
- Commit small, meaningful changes: Easier to review and debug.
- Write clear PR descriptions: Explain the context, not just the code.
- Use automated CI/CD: Test and deploy automatically on each push or merge.
- Delete merged branches: Keep your repo tidy.





