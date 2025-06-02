#!/usr/bin/env python3
"""Dry run script for the inclusion dataset pipeline."""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from inclusion_dataset.config.settings import Config
from inclusion_dataset.pipeline.main import InclusionDatasetPipeline


def load_environment():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        return True
    return False


def create_config() -> Config:
    """Create configuration from environment variables."""
    return Config(
        total_samples=int(os.getenv("TOTAL_SAMPLES", "20000")),
        batch_size=int(os.getenv("BATCH_SIZE", "25")),
        min_quality_score=float(os.getenv("MIN_QUALITY_SCORE", "7.0")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        teacher_model=os.getenv("TEACHER_MODEL", "gpt-4o"),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        output_dir=os.getenv("OUTPUT_DIR", "data/output"),
        log_dir=os.getenv("LOG_DIR", "logs"),
        cache_dir=os.getenv("CACHE_DIR", "data/cache"),
        lexical_diversity_threshold=float(os.getenv("LEXICAL_DIVERSITY_THRESHOLD", "0.4")),
        max_template_overlap=float(os.getenv("MAX_TEMPLATE_OVERLAP", "0.05"))
    )


def print_config_info(config: Config, console: Console):
    """Print configuration information."""
    config_table = Table(title="Pipeline Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="magenta")
    
    config_table.add_row("Teacher Model", config.teacher_model)
    config_table.add_row("Min Quality Score", str(config.min_quality_score))
    config_table.add_row("Batch Size", str(config.batch_size))
    config_table.add_row("Output Directory", config.output_dir)
    config_table.add_row("Max Template Overlap", f"{config.max_template_overlap:.1%}")
    config_table.add_row("Lexical Diversity Threshold", str(config.lexical_diversity_threshold))
    
    console.print(config_table)


def print_results(results: Dict[str, Any], console: Console):
    """Print dry run results."""
    if results["success"]:
        console.print(Panel(
            f"✅ Dry run completed successfully!\n\n"
            f"Samples generated: {results['samples_generated']}\n"
            f"Output file: {results['output_file']}",
            title="Success",
            style="green"
        ))
        
        # Print detailed report
        if "report" in results:
            report = results["report"]
            
            # Summary table
            summary_table = Table(title="Generation Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="magenta")
            
            summary_table.add_row("Total Samples", str(report["total_samples"]))
            
            if "quality_metrics" in report:
                qm = report["quality_metrics"]
                summary_table.add_row("Average Quality Score", f"{qm.get('average_score', 0):.2f}")
                summary_table.add_row("Passed Quality Check", str(qm.get('passed_samples', 0)))
            
            if "diversity_metrics" in report:
                dm = report["diversity_metrics"]
                if "lexical_diversity" in dm:
                    ld = dm["lexical_diversity"]
                    summary_table.add_row("Lexical Diversity (TTR)", f"{ld.get('ttr', 0):.3f}")
                
                if "template_detection" in dm:
                    td = dm["template_detection"]
                    summary_table.add_row("Templates Detected", "❌" if td.get('templates_detected', True) else "✅")
            
            console.print(summary_table)
            
            # Sample distribution
            if "sample_distribution" in report:
                for category, distribution in report["sample_distribution"].items():
                    dist_table = Table(title=f"{category.title()} Distribution")
                    dist_table.add_column("Value", style="cyan")
                    dist_table.add_column("Count", style="magenta")
                    
                    for value, count in distribution.items():
                        dist_table.add_row(str(value), str(count))
                    
                    console.print(dist_table)
            
            # Example samples
            if "sample_examples" in report and report["sample_examples"]:
                console.print(Panel("📝 Sample Examples", style="blue"))
                for i, example in enumerate(report["sample_examples"][:2], 1):
                    console.print(f"\n[bold]Example {i}:[/bold]")
                    console.print(f"[cyan]Instruction:[/cyan] {example.get('instruction', 'N/A')[:100]}...")
                    console.print(f"[cyan]Input:[/cyan] {example.get('input', 'N/A')[:100]}...")
                    console.print(f"[cyan]Output:[/cyan] {example.get('output', 'N/A')[:100]}...")
    else:
        console.print(Panel(
            f"❌ Dry run failed!\n\n"
            f"Error: {results['error']}\n"
            f"Samples generated: {results['samples_generated']}",
            title="Error",
            style="red"
        ))


def main():
    """Main dry run function."""
    parser = argparse.ArgumentParser(description="Run dry test of inclusion dataset pipeline")
    parser.add_argument("--samples", "-s", type=int, default=10, help="Number of samples to generate (default: 10)")
    parser.add_argument("--no-api", action="store_true", help="Run without API calls (mock mode)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    console = Console()
    
    # Load environment
    console.print("🔧 Loading environment...")
    if not load_environment():
        console.print("[yellow]Warning: No .env file found. Using defaults and environment variables.[/yellow]")
    
    # Create configuration
    console.print("⚙️ Creating configuration...")
    try:
        config = create_config()
    except Exception as e:
        console.print(f"[red]Error creating configuration: {e}[/red]")
        return 1
    
    # Check API key
    if not config.openai_api_key and not args.no_api:
        console.print("[red]Error: OPENAI_API_KEY not found. Set it in .env file or use --no-api flag.[/red]")
        return 1
    
    if args.verbose:
        print_config_info(config, console)
    
    # Initialize pipeline
    console.print("🚀 Initializing pipeline...")
    try:
        pipeline = InclusionDatasetPipeline(config)
    except Exception as e:
        console.print(f"[red]Error initializing pipeline: {e}[/red]")
        return 1
    
    # Run dry run
    console.print(f"🔥 Running dry run with {args.samples} samples...")
    
    if args.no_api:
        console.print("[yellow]Note: Running in mock mode (no API calls)[/yellow]")
        # TODO: Implement mock mode
        results = {
            "success": False,
            "error": "Mock mode not yet implemented",
            "samples_generated": 0
        }
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating samples...", total=None)
            
            try:
                results = pipeline.run_dry_run(sample_size=args.samples)
            except Exception as e:
                console.print(f"[red]Error during dry run: {e}[/red]")
                if args.verbose:
                    import traceback
                    console.print(traceback.format_exc())
                return 1
            finally:
                progress.remove_task(task)
    
    # Print results
    print_results(results, console)
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())