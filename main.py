#!/usr/bin/env python3
"""Main CLI entry point for Edge-Cloud Code Analysis."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from shared.autodl_env import apply_autodl_data_disk_env

apply_autodl_data_disk_env()

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import structlog

from core.orchestrator import Orchestrator
from shared.schemas import CodeLanguage

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

console = Console()


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dict
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load environment variables for API keys
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # Substitute environment variables
        if 'cloud' in config:
            for provider in config['cloud']:
                if isinstance(config['cloud'][provider], dict):
                    api_key = config['cloud'][provider].get('api_key', '')
                    if api_key.startswith('${') and api_key.endswith('}'):
                        env_var = api_key[2:-1]
                        config['cloud'][provider]['api_key'] = os.getenv(env_var, '')
        
        return config
    
    except FileNotFoundError:
        console.print(f"[red]Error: Configuration file not found: {config_path}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)


@click.group()
@click.option('--config', default='config/settings.yaml', help='Config file path')
@click.pass_context
def cli(ctx, config):
    """Edge-Cloud Code Analysis Tool.
    
    Analyze code using local LLM (Ollama) with selective cloud verification.
    """
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)


@cli.command()
@click.option('--file', '-f', type=str, help='Single file to analyze')
@click.option('--dir', '-d', type=str, help='Directory to analyze')
@click.option('--pattern', default='**/*.py', help='File pattern for directory scan')
@click.option('--language', '-l', type=click.Choice(['python', 'java', 'javascript', 'cpp']), 
              help='Programming language')
@click.option('--max-cloud-calls', type=int, help='Maximum cloud API calls')
@click.option('--output', '-o', type=str, help='Output file for results')
@click.pass_context
def analyze(ctx, file, dir, pattern, language, max_cloud_calls, output):
    """Analyze code files for potential bugs."""
    config = ctx.obj['config']
    
    if not file and not dir:
        console.print("[red]Error: Must specify either --file or --dir[/red]")
        sys.exit(1)
    
    # Override max cloud calls if specified
    if max_cloud_calls:
        config.setdefault('strategy', {})['max_concurrent_cloud_calls'] = max_cloud_calls
    
    # Convert language string to enum
    lang = None
    if language:
        lang = CodeLanguage(language)
    
    # Run analysis
    asyncio.run(run_analysis(config, file, dir, pattern, lang, output))


async def run_analysis(
    config: dict,
    file_path: Optional[str],
    dir_path: Optional[str],
    pattern: str,
    language: Optional[CodeLanguage],
    output_path: Optional[str]
):
    """Run the analysis workflow."""
    console.print(Panel.fit(
        "[bold cyan]Edge-Cloud Code Analysis[/bold cyan]\n"
        "Local: Ollama | Cloud: Multi-Provider",
        border_style="cyan"
    ))
    
    # Initialize orchestrator
    orchestrator = Orchestrator(config)
    
    try:
        # Run analysis
        if file_path:
            console.print(f"\n[yellow]Analyzing file:[/yellow] {file_path}")
            results_dict = {file_path: await orchestrator.analyze_file(file_path, language)}
        else:
            console.print(f"\n[yellow]Analyzing directory:[/yellow] {dir_path}")
            console.print(f"[yellow]Pattern:[/yellow] {pattern}")
            results_dict = await orchestrator.analyze_directory(dir_path, pattern)
        
        # Display results
        display_results(results_dict)
        
        # Show metrics
        await display_metrics(orchestrator)
        
        # Export results if requested
        if output_path:
            export_results(results_dict, output_path)
            console.print(f"\n[green]✓ Results exported to {output_path}[/green]")
    
    finally:
        await orchestrator.shutdown()


def display_results(results_dict: dict):
    """Display analysis results in a nice table."""
    total_issues = sum(len(results) for results in results_dict.values())
    
    if total_issues == 0:
        console.print("\n[green]✓ No issues found![/green]")
        return
    
    console.print(f"\n[bold]Found {total_issues} potential issues:[/bold]\n")
    
    for file_path, results in results_dict.items():
        if not results:
            continue
        
        console.print(f"[cyan]{file_path}[/cyan]")
        
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Line", style="dim", width=8)
        table.add_column("Severity", width=10)
        table.add_column("Type", width=12)
        table.add_column("Description", width=40)
        table.add_column("Confidence", width=10)
        table.add_column("Verified", width=10)
        
        for result in results:
            # Color code severity
            severity_colors = {
                'critical': '[red]',
                'high': '[yellow]',
                'medium': '[blue]',
                'low': '[green]',
                'info': '[dim]'
            }
            severity_color = severity_colors.get(result.draft.severity.value, '')
            severity_text = f"{severity_color}{result.draft.severity.value}[/]"
            
            # Confidence indicator
            conf_pct = f"{result.final_confidence:.0%}"
            
            # Verified status
            verified = "✓ Yes" if result.was_verified else "Local"
            
            table.add_row(
                f"{result.draft.fragment.start_line}",
                severity_text,
                result.draft.issue_type.value,
                result.final_description[:40] + "..." if len(result.final_description) > 40 else result.final_description,
                conf_pct,
                verified
            )
        
        console.print(table)
        console.print()


async def display_metrics(orchestrator: Orchestrator):
    """Display analysis metrics and budget status."""
    metrics = await orchestrator.get_metrics()
    budget = await orchestrator.get_budget_status()
    
    # Metrics table
    console.print("[bold]Analysis Metrics:[/bold]")
    metrics_table = Table(box=box.SIMPLE, show_header=False)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    metrics_table.add_row("Total Fragments", str(metrics.total_fragments))
    metrics_table.add_row("Local Only", str(metrics.local_only))
    metrics_table.add_row("Cloud Verified", str(metrics.cloud_verified))
    
    if metrics.total_fragments > 0:
        cloud_ratio = metrics.cloud_verified / metrics.total_fragments
        metrics_table.add_row("Cloud Upload Rate", f"{cloud_ratio:.1%}")
    
    console.print(metrics_table)
    console.print()
    
    # Budget status
    console.print("[bold]Budget Status:[/bold]")
    budget_table = Table(box=box.SIMPLE, show_header=False)
    budget_table.add_column("Item", style="cyan")
    budget_table.add_column("Value", style="yellow")
    
    budget_table.add_row("Total Budget", f"${budget.total_budget:.2f}")
    budget_table.add_row("Used", f"${budget.used_budget:.2f}")
    budget_table.add_row("Remaining", f"${budget.remaining_budget:.2f}")
    budget_table.add_row("Remaining %", f"{budget.remaining_percent:.1%}")
    
    # Color code based on remaining
    if budget.remaining_percent < 0.2:
        budget_status_color = "red"
        status_msg = "⚠️  CRITICAL"
    elif budget.remaining_percent < 0.4:
        budget_status_color = "yellow"
        status_msg = "⚠️  LOW"
    else:
        budget_status_color = "green"
        status_msg = "✓ GOOD"
    
    budget_table.add_row("Status", f"[{budget_status_color}]{status_msg}[/]")
    
    console.print(budget_table)


def export_results(results_dict: dict, output_path: str):
    """Export results to JSON file."""
    import json
    from datetime import datetime
    
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'files': {}
    }
    
    for file_path, results in results_dict.items():
        export_data['files'][file_path] = [
            {
                'location': result.location,
                'severity': result.draft.severity.value,
                'type': result.draft.issue_type.value,
                'description': result.final_description,
                'confidence': result.final_confidence,
                'suggested_fix': result.final_fix,
                'was_verified': result.was_verified
            }
            for result in results
        ]
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)


@cli.command()
@click.pass_context
def budget_status(ctx):
    """Show current budget status."""
    config = ctx.obj['config']
    asyncio.run(show_budget_status(config))


async def show_budget_status(config: dict):
    """Display detailed budget status."""
    orchestrator = Orchestrator(config)
    await orchestrator.initialize()
    
    try:
        budget = await orchestrator.get_budget_status()
        
        console.print(Panel.fit("[bold cyan]Budget Status (In-Memory)[/bold cyan]", border_style="cyan"))
        
        # Current status
        console.print("\n[bold]Current Status:[/bold]")
        console.print(f"  Total Budget: ${budget.total_budget:.2f}")
        console.print(f"  Used: ${budget.used_budget:.2f}")
        console.print(f"  Remaining: ${budget.remaining_budget:.2f} ({budget.remaining_percent:.1%})")
        
        console.print("\n[yellow]Note: Budget tracking is now in-memory only (no database persistence).[/yellow]")
    
    finally:
        await orchestrator.shutdown()


@cli.command()
@click.pass_context
def health_check(ctx):
    """Check health of all configured providers."""
    config = ctx.obj['config']
    asyncio.run(run_health_check(config))


async def run_health_check(config: dict):
    """Run health check on all providers."""
    from cloud.provider_factory import ProviderFactory
    
    console.print(Panel.fit("[bold cyan]Provider Health Check[/bold cyan]", border_style="cyan"))
    
    factory = ProviderFactory(config.get('cloud', {}))
    health_status = await factory.check_all_health()
    
    console.print("\n[bold]Provider Status:[/bold]\n")
    
    for provider, is_healthy in health_status.items():
        status = "[green]✓ Healthy[/green]" if is_healthy else "[red]✗ Unavailable[/red]"
        console.print(f"  {provider}: {status}")


if __name__ == '__main__':
    cli(obj={})

