"""
Phase 6: Architect Agent — Architecture design, component decomposition,
         interface design, design pattern application.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .code_understanding import RepositoryAnalyzer, Symbol


@dataclass
class Component:
    name: str
    kind: str  # module, service, library, data_model, interface, utility
    description: str = ""
    responsibility: str = ""
    depends_on: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    public_interface: list[str] = field(default_factory=list)


@dataclass
class Architecture:
    name: str
    description: str = ""
    components: list[Component] = field(default_factory=list)
    data_flow: str = ""
    patterns: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


class ArchitectAgent:
    """Analyzes repository structure and proposes architecture improvements."""
    
    def __init__(self, analyzer: RepositoryAnalyzer, provider=None):
        self.analyzer = analyzer
        self.provider = provider  # LLM provider for intelligent suggestions
    
    def analyze_current_architecture(self) -> Architecture:
        """Extract current architecture from code structure."""
        arch = Architecture(name="Current Architecture")
        components: dict[str, Component] = {}
        
        for fpath, parser in self.analyzer.parsers.items():
            # Determine component from directory structure
            rel_path = Path(fpath).relative_to(self.analyzer.root)
            comp_name = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
            
            if comp_name not in components:
                components[comp_name] = Component(
                    name=comp_name,
                    kind=self._infer_kind(comp_name, parser),
                    files=[],
                )
            
            comp = components[comp_name]
            comp.files.append(str(rel_path))
            comp.public_interface.extend(s.name for s in parser.symbols if s.visibility == "public")
        
        # Resolve dependencies
        for fpath, deps in self.analyzer.dep_graph.items():
            rel = Path(fpath).relative_to(self.analyzer.root)
            src_comp = rel.parts[0] if len(rel.parts) > 1 else "root"
            for dep in deps:
                dep_rel = Path(dep.target_file).relative_to(self.analyzer.root) if dep.target_file else None
                if dep_rel:
                    tgt_comp = dep_rel.parts[0] if len(dep_rel.parts) > 1 else "root"
                    if src_comp in components and tgt_comp in components:
                        if tgt_comp not in components[src_comp].depends_on:
                            components[src_comp].depends_on.append(tgt_comp)
        
        arch.components = list(components.values())
        arch.patterns = self._detect_patterns()
        arch.risks = self._detect_risks(arch)
        
        return arch
    
    def _infer_kind(self, name: str, parser) -> str:
        name_lower = name.lower()
        if any(k in name_lower for k in ('test', 'spec')):
            return "test"
        if any(k in name_lower for k in ('model', 'entity', 'schema', 'dto')):
            return "data_model"
        if any(k in name_lower for k in ('api', 'controller', 'handler', 'router', 'view')):
            return "interface"
        if any(k in name_lower for k in ('service', 'usecase', 'domain')):
            return "service"
        if any(k in name_lower for k in ('repo', 'db', 'database', 'store', 'cache')):
            return "data_access"
        if any(k in name_lower for k in ('util', 'helper', 'common', 'shared')):
            return "utility"
        return "module"
    
    def _detect_patterns(self) -> list[str]:
        """Detect design/architectural patterns from code structure."""
        patterns = []
        
        # Check for MVC
        has_model = any(c.kind == "data_model" for c in self.analyze_current_architecture().components)
        has_controller = any(c.kind == "interface" for c in self.analyze_current_architecture().components)
        if has_model and has_controller:
            patterns.append("MVC-like")
        
        # Check for repository pattern
        has_repo = any(c.kind == "data_access" for c in self.analyze_current_architecture().components)
        if has_repo:
            patterns.append("Repository")
        
        # Check for factory
        for parser in self.analyzer.parsers.values():
            for sym in parser.symbols:
                if sym.kind == "function" and ("factory" in sym.name.lower() or "create" in sym.name.lower()):
                    patterns.append("Factory")
                    break
            if "Factory" in patterns:
                break
        
        # Microservice detection (many independent components)
        n_components = len(set(
            Path(f).relative_to(self.analyzer.root).parts[0]
            for f in self.analyzer.parsers
            if Path(f).relative_to(self.analyzer.root).parts
        ))
        if n_components >= 5:
            patterns.append("Modular/Microservice-like")
        
        return sorted(set(patterns))
    
    def _detect_risks(self, arch: Architecture) -> list[str]:
        """Detect architectural risks."""
        risks = []
        
        for comp in arch.components:
            # Too many files in one component
            if len(comp.files) > 50:
                risks.append(f"Large component '{comp.name}': {len(comp.files)} files — consider splitting")
            
            # No public interface
            if not comp.public_interface and comp.files:
                risks.append(f"No public symbols detected in '{comp.name}' — encapsulation issue?")
            
            # Circular dependency risk (detected via mutual depends_on)
            for other in arch.components:
                if (comp.name in other.depends_on and other.name in comp.depends_on):
                    risks.append(f"Potential circular dependency: {comp.name} ↔ {other.name}")
        
        # Single large component risk
        if len(arch.components) == 1 and len(arch.components[0].files) > 20:
            risks.append("Single monolithic structure — consider modular decomposition")
        
        return risks
    
    def suggest_improvements(self) -> list[dict]:
        """Propose architecture improvements."""
        arch = self.analyze_current_architecture()
        suggestions = []
        
        # 1. Large component splitting
        for comp in arch.components:
            if len(comp.files) > 30:
                suggestions.append({
                    "type": "decompose",
                    "component": comp.name,
                    "reason": f"Too many files ({len(comp.files)}). Consider splitting by responsibility.",
                    "proposed_modules": self._suggest_split(comp),
                })
        
        # 2. Missing patterns
        has_service_layer = any(c.kind == "service" for c in arch.components)
        has_interface_layer = any(c.kind == "interface" for c in arch.components)
        if not has_service_layer and len(self.analyzer.parsers) > 10:
            suggestions.append({
                "type": "add_layer",
                "reason": "Consider adding a service/business logic layer to separate concerns.",
                "proposed": "services/",
            })
        if not has_interface_layer and len(self.analyzer.parsers) > 10:
            suggestions.append({
                "type": "add_layer",
                "reason": "Consider adding an interface/API layer for clean separation.",
                "proposed": "api/ or controllers/",
            })
        
        # 3. Circular dependencies
        for risk in arch.risks:
            if "circular" in risk.lower():
                suggestions.append({
                    "type": "fix_circular",
                    "reason": risk,
                    "proposed": "Extract shared interface or use dependency inversion.",
                })
        
        return suggestions
    
    def _suggest_split(self, comp: Component) -> list[str]:
        """Suggest how to split a large component."""
        # Group files by subdirectory
        by_dir: dict[str, int] = {}
        for f in comp.files:
            parts = Path(f).parts
            sub = parts[1] if len(parts) > 2 else "core"
            by_dir[sub] = by_dir.get(sub, 0) + 1
        
        # Suggest directories with enough files
        return [f"{comp.name}/{d}" for d, cnt in sorted(by_dir.items(), key=lambda x: -x[1])[:4] if cnt >= 3]
    
    def architecture_report(self) -> str:
        """Generate a comprehensive architecture report."""
        arch = self.analyze_current_architecture()
        metrics = self.analyzer.metrics
        suggestions = self.suggest_improvements()
        
        lines = []
        lines.append("# Architecture Report")
        lines.append(f"\n## Overview")
        lines.append(f"- Files: {metrics.get('total_files', 0)}")
        lines.append(f"- Symbols: {metrics.get('total_symbols', 0)}")
        lines.append(f"- Functions: {metrics.get('total_functions', 0)}")
        lines.append(f"- Classes: {metrics.get('total_classes', 0)}")
        
        lines.append(f"\n## Components ({len(arch.components)})")
        for comp in arch.components:
            lines.append(f"\n### {comp.name} ({comp.kind})")
            lines.append(f"- Files: {len(comp.files)}")
            if comp.depends_on:
                lines.append(f"- Depends on: {', '.join(comp.depends_on[:10])}")
            if comp.public_interface:
                lines.append(f"- Public API: {', '.join(comp.public_interface[:10])}")
        
        lines.append(f"\n## Patterns Detected")
        for p in arch.patterns:
            lines.append(f"- {p}")
        
        lines.append(f"\n## Risks ({len(arch.risks)})")
        for r in arch.risks:
            lines.append(f"- ⚠ {r}")
        
        lines.append(f"\n## Improvement Suggestions ({len(suggestions)})")
        for s in suggestions:
            lines.append(f"- [{s['type']}] {s['reason']}")
        
        return "\n".join(lines)
