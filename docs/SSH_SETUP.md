# SSH Key Setup Guide for GitHub (Windows)

## Step 1: Check for Existing SSH Keys

First, check if you already have SSH keys:

```powershell
# Check for existing SSH keys
ls ~/.ssh
```

If you see files like `id_rsa` and `id_rsa.pub` (or `id_ed25519` and `id_ed25519.pub`), you already have SSH keys. Skip to Step 3.

## Step 2: Generate a New SSH Key

If you don't have SSH keys, generate a new one:

```powershell
# Generate SSH key (replace with your GitHub email)
ssh-keygen -t ed25519 -C "your_email@example.com"

# If ed25519 is not supported, use RSA instead:
# ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

**When prompted:**
- **File location**: Press Enter to accept the default (`C:\Users\YOUR_USERNAME\.ssh\id_ed25519`)
- **Passphrase**: You can set a passphrase for extra security (recommended) or press Enter twice to skip

## Step 3: Start the SSH Agent

```powershell
# Start the ssh-agent service
Get-Service ssh-agent | Set-Service -StartupType Automatic
Start-Service ssh-agent

# Or if the above doesn't work:
# Start-Service ssh-agent
```

## Step 4: Add Your SSH Key to the SSH Agent

```powershell
# Add your SSH private key to the ssh-agent
ssh-add ~/.ssh/id_ed25519

# If you used RSA instead:
# ssh-add ~/.ssh/id_rsa
```

## Step 5: Copy Your Public SSH Key

```powershell
# Display your public key (copy the entire output)
cat ~/.ssh/id_ed25519.pub

# Or if you used RSA:
# cat ~/.ssh/id_rsa.pub
```

**Important**: Copy the entire output (it starts with `ssh-ed25519` or `ssh-rsa` and ends with your email).

## Step 6: Add SSH Key to GitHub

1. Go to [GitHub.com](https://github.com) and sign in
2. Click your profile picture → **Settings**
3. In the left sidebar, click **SSH and GPG keys**
4. Click **New SSH key** or **Add SSH key**
5. In the "Title" field, add a descriptive label (e.g., "Windows Laptop")
6. In the "Key" field, paste your public key (the one you copied in Step 5)
7. Click **Add SSH key**
8. If prompted, confirm your GitHub password

## Step 7: Test Your SSH Connection

```powershell
# Test the connection to GitHub
ssh -T git@github.com
```

**Expected output:**
- If successful: `Hi USERNAME! You've successfully authenticated, but GitHub does not provide shell access.`
- If it asks to add to known hosts: Type `yes` and press Enter

## Troubleshooting

### If you get "Permission denied (publickey)":

1. **Verify the key was added to ssh-agent:**
   ```powershell
   ssh-add -l
   ```
   Should show your key. If not, add it again with `ssh-add ~/.ssh/id_ed25519`

2. **Verify the key is on GitHub:**
   - Go to GitHub Settings → SSH and GPG keys
   - Make sure your key is listed there

3. **Try using the full path:**
   ```powershell
   ssh-add C:\Users\YOUR_USERNAME\.ssh\id_ed25519
   ```

### If ssh-agent doesn't start:

```powershell
# Try running as Administrator
Set-Service -Name ssh-agent -StartupType 'Automatic'
Start-Service ssh-agent
```

### If you need to use a different key:

```powershell
# Create/edit SSH config file
notepad ~/.ssh/config
```

Add:
```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
```

## Next Steps

Once SSH is configured, you can use SSH URLs for your Git remotes:

```powershell
git remote add origin git@github.com:YOUR_USERNAME/credit-risk-ml-pipeline.git
```
