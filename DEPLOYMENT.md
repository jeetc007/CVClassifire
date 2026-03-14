# Deployment Guide for Customer Classification App

## 🚀 Deploying to Streamlit Cloud

This guide will help you deploy your Customer Classification ML project to Streamlit Cloud.

### Prerequisites
- GitHub account
- Streamlit Cloud account (sign up at [share.streamlit.io](https://share.streamlit.io))
- Trained model artifacts in `artifacts/models/`
- Processed data in `data/processed/`

### Step 1: Prepare Your Repository

1. **Ensure all artifacts are committed:**
   ```bash
   git add artifacts/models/*.pkl
   git add data/processed/*.csv
   git commit -m "Add model artifacts and processed data"
   ```

2. **Push to GitHub:**
   ```bash
   git push origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud:** [share.streamlit.io](https://share.streamlit.io)

2. **Connect your GitHub account**

3. **Create new app:**
   - Click "New app"
   - Select your GitHub repository
   - Select the main branch
   - Set the main file path to: `app/app.py`
   - Click "Deploy"

4. **Advanced configuration (optional):**
   - Add environment variables if needed
   - Configure secrets if using API keys
   - Set custom Python version if required

### Step 3: Verify Deployment

Once deployed, your app will be available at:
`https://your-username-your-repo.streamlit.app`

### 🔧 Deployment Configuration Files

The project includes:
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration
- `README.md` - Project documentation

### 📋 Deployment Checklist

- [ ] Model training completed and artifacts saved
- [ ] All dependencies listed in requirements.txt with versions
- [ ] App runs locally without errors
- [ ] Repository pushed to GitHub
- [ ] No hardcoded file paths (use relative paths)
- [ ] Error handling implemented
- [ ] Memory usage optimized for cloud environment

### 🐛 Common Deployment Issues

1. **Model file not found**
   - Ensure `artifacts/models/` directory is committed to Git
   - Check file paths in the code

2. **Memory limits**
   - Streamlit Cloud has memory limitations
   - Consider optimizing large data files

3. **Dependency conflicts**
   - Use specific versions in requirements.txt
   - Test with clean environment

4. **Long loading times**
   - Optimize data loading
   - Consider caching strategies

### 🔄 Continuous Deployment

For automatic updates when you push to GitHub:
- Enable auto-deploy in Streamlit Cloud settings
- Ensure your main branch is always stable

### 📊 Monitoring

Streamlit Cloud provides:
- App usage analytics
- Error logs
- Performance metrics
- Resource usage monitoring

## 🎯 Next Steps

After deployment:
1. Share your app with stakeholders
2. Monitor performance and usage
3. Gather feedback for improvements
4. Plan for scaling if needed

## 📞 Support

For deployment issues:
- Check Streamlit Cloud documentation
- Review error logs in the dashboard
- Ensure all files are properly committed to Git
