# Contributing to EEG/BCI Research Copilot

Thank you for your interest in contributing to the EEG/BCI Research Copilot! This project aims to make neuroscience research more accessible through AI-powered literature analysis.

## 📄 License

This project is licensed under the Apache License 2.0. This allows for broad use and modification while maintaining attribution requirements and patent protections. See the [LICENSE](LICENSE) file for details.

## 🚀 Quick Start

1. **Fork and clone** the repository
2. **Set up your environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Configure API keys**:
   ```bash
   cp .env.example .env
   # Edit .env with your Gemini API key
   ```
4. **Run the app**:
   ```bash
   python app.py
   ```

## 🛠️ Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines when possible

### Testing
- Test your changes locally before submitting
- Ensure the Gradio interface loads without errors
- Verify that paper ingestion and chat functionality work
- Check that example queries in the Research Chat tab produce valid responses

### Pull Request Process
1. **Create a feature branch** from `main`
2. **Make your changes** with clear, descriptive commit messages
3. **Test thoroughly** — the app should work end-to-end
4. **Update documentation** if needed (README.md, docstrings)
5. **Submit a pull request** with a detailed description of your changes

## 📋 Areas for Contribution

### 🔬 Research Features
- Add new agent tools for specific research tasks
- Improve paper summarization and method extraction
- Enhance dataset comparison capabilities
- Add support for new document formats (beyond PDF)

### 🏗️ Technical Improvements
- Optimize vector search performance
- Add more robust error handling
- Implement caching for repeated queries
- Add unit tests and integration tests

### 📊 UI/UX Enhancements
- Improve the Gradio interface design
- Add more example queries and use cases
- Enhance source citation display
- Add export functionality for chat history

### 📚 Documentation
- Expand the README with more examples
- Add tutorials for specific research workflows
- Create video demonstrations
- Translate documentation to other languages

## 🐛 Bug Reports

If you find a bug, please:
1. Check if it's already reported in the Issues
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs. actual behavior
   - Your environment (OS, Python version, etc.)

## 💡 Feature Requests

We welcome feature ideas! Please:
1. Check existing Issues for similar requests
2. Create a new Issue with the `enhancement` label
3. Describe the feature and its benefits
4. Explain how it fits into the research workflow

## 📞 Contact

For questions or discussions:
- Open an Issue on GitHub
- Email: khunsaiftikhar123@gmail.com

Thank you for helping advance neuroscience research! 🧠✨