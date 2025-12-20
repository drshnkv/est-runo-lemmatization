#!/usr/bin/env python3
"""
Configuration Manager for Estonian Runosong Lemmatizer Scripts

Handles secure storage and retrieval of API keys and other configuration settings.
"""

import json
import os
import getpass
from pathlib import Path
from typing import Optional, Dict, Any
import logging

class ConfigManager:
    """
    Manages configuration settings including secure API key storage.
    """
    
    def __init__(self, config_dir: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to store config files (default: ~/.runosong_lemmatizer)
        """
        if config_dir is None:
            self.config_dir = Path.home() / ".runosong_lemmatizer"
        else:
            self.config_dir = Path(config_dir)
        
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        self.api_key_file = self.config_dir / ".api_key"
        
        # Set secure permissions on config directory
        try:
            os.chmod(self.config_dir, 0o700)  # Owner read/write/execute only
        except:
            pass  # Ignore on Windows
        
        self.logger = logging.getLogger(__name__)
    
    def save_api_key(self, api_key: str) -> None:
        """
        Securely save API key to file.
        
        Args:
            api_key: Anthropic API key to save
        """
        try:
            with open(self.api_key_file, 'w') as f:
                f.write(api_key)
            
            # Set secure permissions
            try:
                os.chmod(self.api_key_file, 0o600)  # Owner read/write only
            except:
                pass  # Ignore on Windows
            
            self.logger.info("API key saved successfully")
            print("✅ API key saved successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to save API key: {e}")
            raise
    
    def get_api_key(self) -> Optional[str]:
        """
        Retrieve saved API key.
        
        Returns:
            API key if found, None otherwise
        """
        # First check environment variable
        env_key = os.environ.get('ANTHROPIC_API_KEY')
        if env_key:
            return env_key
        
        # Then check saved file
        if self.api_key_file.exists():
            try:
                with open(self.api_key_file, 'r') as f:
                    api_key = f.read().strip()
                    if api_key:
                        return api_key
            except Exception as e:
                self.logger.warning(f"Failed to read saved API key: {e}")
        
        return None
    
    def prompt_for_api_key(self) -> str:
        """
        Prompt user for API key and optionally save it.
        
        Returns:
            API key entered by user
        """
        print("\n🔑 Anthropic API Key Required")
        print("You can get your API key from: https://console.anthropic.com/")
        
        api_key = getpass.getpass("Enter your Anthropic API key: ").strip()
        
        if not api_key:
            raise ValueError("API key cannot be empty")
        
        # Validate API key format (basic check)
        if not api_key.startswith('sk-ant-'):
            print("⚠️  Warning: API key doesn't match expected format (should start with 'sk-ant-')")
        
        # Ask if user wants to save the key
        save_choice = input("\n💾 Save API key for future use? (y/n): ").lower().strip()
        if save_choice in ['y', 'yes']:
            try:
                self.save_api_key(api_key)
            except Exception as e:
                print(f"⚠️  Could not save API key: {e}")
                print("You'll need to enter it again next time.")
        
        return api_key
    
    def ensure_api_key(self) -> str:
        """
        Ensure API key is available, prompting if necessary.
        
        Returns:
            Valid API key
        """
        # Try to get existing key
        api_key = self.get_api_key()
        
        if api_key:
            return api_key
        
        # No key found, prompt user
        return self.prompt_for_api_key()
    
    def delete_api_key(self) -> None:
        """Delete saved API key."""
        try:
            if self.api_key_file.exists():
                self.api_key_file.unlink()
                print("🗑️  API key deleted successfully")
            else:
                print("ℹ️  No saved API key found")
        except Exception as e:
            print(f"❌ Failed to delete API key: {e}")
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration settings.
        
        Args:
            config: Configuration dictionary to save
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info("Configuration saved")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration settings.
        
        Returns:
            Configuration dictionary
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load configuration: {e}")
        
        # Return default configuration
        return {
            "annotation_dir": "annotation_converter_output_data/",
            "confidence_threshold": 0.6,
            "text_column": "text",
            "title_column": "poemTitle",
            "processing_options": {
                "enable_ai_enhancement": True,
                "save_intermediate": True,
                "auto_save_interval": 5
            }
        }
    
    def get_annotation_dir(self) -> str:
        """Get the default annotation directory from config."""
        config = self.load_config()
        return config.get("annotation_dir", "annotation_converter_output_data/")
    
    def setup_wizard(self) -> None:
        """
        Interactive setup wizard for first-time configuration.
        """
        print("\n🎵 Estonian Runosong Lemmatizer - Setup Wizard")
        print("=" * 50)
        
        # Get API key
        print("\n1. API Key Setup")
        api_key = self.ensure_api_key()
        
        # Get annotation directory
        print("\n2. Annotation Directory Setup")
        default_dir = "annotation_converter_output_data/"
        annotation_dir = input(f"Annotation directory (default: {default_dir}): ").strip()
        if not annotation_dir:
            annotation_dir = default_dir
        
        # Verify annotation directory exists
        if not Path(annotation_dir).exists():
            print(f"⚠️  Directory '{annotation_dir}' does not exist")
            create_choice = input("Create directory? (y/n): ").lower().strip()
            if create_choice in ['y', 'yes']:
                Path(annotation_dir).mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {annotation_dir}")
            else:
                print("Please create the directory manually or use a different path")
                return
        
        # Save configuration
        config = {
            "annotation_dir": annotation_dir,
            "confidence_threshold": 0.6,
            "text_column": "text", 
            "title_column": "poemTitle",
            "processing_options": {
                "enable_ai_enhancement": True,
                "save_intermediate": True,
                "auto_save_interval": 5
            },
            "setup_completed": True,
            "setup_date": str(Path().cwd())  # Save current directory as reference
        }
        
        self.save_config(config)
        
        print("\n✅ Setup completed successfully!")
        print(f"📁 Annotation directory: {annotation_dir}")
        print(f"🔑 API key: {'Saved' if api_key else 'Not saved'}")
        print("\nYou can now run the lemmatizer scripts without specifying --api-key or --annotation-dir")
        print("\nExample usage:")
        print("  python ai_primary_runosong_lemmatizer.py --text 'Your runosong text here'")
        print("  python claude_enhanced_lemmatizer.py --input runosongs.csv --output results.csv")


def main():
    """Command-line interface for configuration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration Manager for Runosong Lemmatizer')
    parser.add_argument('--setup', action='store_true', help='Run setup wizard')
    parser.add_argument('--save-key', help='Save API key')
    parser.add_argument('--delete-key', action='store_true', help='Delete saved API key')
    parser.add_argument('--show-config', action='store_true', help='Show current configuration')
    parser.add_argument('--reset', action='store_true', help='Reset all configuration')
    
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    
    if args.setup:
        config_manager.setup_wizard()
    elif args.save_key:
        config_manager.save_api_key(args.save_key)
    elif args.delete_key:
        config_manager.delete_api_key()
    elif args.show_config:
        config = config_manager.load_config()
        api_key_status = "✅ Saved" if config_manager.get_api_key() else "❌ Not found"
        print(f"\n📋 Current Configuration:")
        print(f"🔑 API Key: {api_key_status}")
        print(f"📁 Annotation Directory: {config.get('annotation_dir', 'Not set')}")
        print(f"🎯 Confidence Threshold: {config.get('confidence_threshold', 'Not set')}")
        print(f"📊 Processing Options: {config.get('processing_options', 'Default')}")
    elif args.reset:
        confirm = input("⚠️  This will delete all saved configuration. Continue? (y/n): ")
        if confirm.lower() in ['y', 'yes']:
            config_manager.delete_api_key()
            if config_manager.config_file.exists():
                config_manager.config_file.unlink()
            print("✅ Configuration reset successfully")
        else:
            print("Reset cancelled")
    else:
        print("No action specified. Use --help for options.")
        print("\nQuick setup: python config_manager.py --setup")


if __name__ == "__main__":
    main()