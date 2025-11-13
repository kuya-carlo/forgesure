"""
Core business logic for steel plate optimization.

This module implements the PlateOptimizationService class which encapsulates
all operations related to plate data processing, KPI calculations, and linear
programming optimization using OOP principles.
"""

from typing import List, Dict, Tuple
import pandas as pd
import pulp
from models import (
    PlateRequest,
    OptimizationConstraints,
    OptimizationWeights,
    PlateKPIResponse,
    OptimizationResponse
)


class PlateDataProcessor:
    """
    Handles data ingestion and KPI calculation for steel plates.
    
    Responsible for:
    - Converting raw plate data to DataFrame
    - Calculating geometric KPIs (area, volume, weight)
    - Computing energy consumption metrics
    - Estimating material, energy, and labor costs
    """

    STEEL_DENSITY = 7.85  # ton/m³
    PROCESS_TIME_COLUMNS = [
        'cutting_time', 'bevelling_time', 'drilling_time',
        'splice_welding_time', 'saw_welding_time', 'straightening_time',
        'accessories_time', 'grinding_time', 'shot_blasting_time', 'painting_time'
    ]
    ENERGY_MULTIPLIER = 10  # kWh per hour

    def __init__(self, plates: List[PlateRequest], constraints: OptimizationConstraints):
        """
        Initialize the processor with plate data and constraints.
        
        Args:
            plates: List of PlateRequest objects
            constraints: OptimizationConstraints with cost parameters
        """
        self.plates = plates
        self.constraints = constraints
        self.df = None

    def process_plates(self) -> pd.DataFrame:
        """
        Transform raw plate data into enriched DataFrame with all KPIs.
        
        Returns:
            DataFrame containing original data plus calculated KPIs
        """
        # Convert to DataFrame
        plates_data = [
            {
                'id': p.id,
                'thickness': p.thickness,
                'length': p.length,
                'width': p.width,
                'holes': p.holes,
                'beam_size': p.beam_size,
                'price': p.price
            }
            for p in self.plates
        ]
        self.df = pd.DataFrame(plates_data)

        # Add process times (simplified linear relationship based on size)
        self._calculate_process_times()

        # Calculate geometric KPIs
        self._calculate_geometric_kpis()

        # Calculate energy metrics
        self._calculate_energy_metrics()

        # Calculate costs
        self._calculate_costs()

        # Calculate profit
        self._calculate_profit()

        return self.df

    def _calculate_process_times(self):
        """Calculate process times for each manufacturing step."""
        # Base times depend on plate complexity (thickness * length * width factor)
        complexity = (self.df['thickness'] * self.df['length'] * self.df['width']) / 1e8

        self.df['cutting_time'] = complexity * 1.5
        self.df['bevelling_time'] = complexity
        self.df['drilling_time'] = self.df['holes'] * 0.05 + complexity * 0.2
        self.df['splice_welding_time'] = complexity * 1.2
        self.df['saw_welding_time'] = complexity * 0.6
        self.df['straightening_time'] = complexity * 0.4
        self.df['accessories_time'] = complexity * 0.7
        self.df['grinding_time'] = complexity * 0.3
        self.df['shot_blasting_time'] = complexity * 0.5
        self.df['painting_time'] = complexity * 2

    def _calculate_geometric_kpis(self):
        """Calculate geometric properties and utilization metrics."""
        self.df['plate_area'] = self.df['length'] * self.df['width']
        self.df['net_parts_area'] = self.df['thickness'] * self.df['length'] * self.df['width']
        self.df['nesting_utilization'] = (self.df['net_parts_area'] / self.df['plate_area']).clip(0, 1)

        # Volume and weight calculations
        self.df['beam_volume_m3'] = (
            self.df['length'] / 1000 * self.df['width'] / 1000 * self.df['thickness'] / 1000
        )
        self.df['beam_weight_ton'] = self.df['beam_volume_m3'] * self.STEEL_DENSITY

    def _calculate_energy_metrics(self):
        """Calculate total energy consumption and intensity."""
        self.df['energy_total_kWh'] = (
            self.df[self.PROCESS_TIME_COLUMNS].sum(axis=1) * self.ENERGY_MULTIPLIER
        )
        self.df['energy_intensity_kWh_per_ton'] = (
            self.df['energy_total_kWh'] / self.df['beam_weight_ton']
        )
        # First-pass yield: larger plates have slightly lower yield
        self.df['first_pass_yield'] = 1.0 - (self.df['holes'] * 0.005).clip(0, 0.1)

    def _calculate_costs(self):
        """Calculate material, energy, and labor costs."""
        self.df['material_cost'] = (
            self.df['beam_weight_ton'] * self.constraints.material_cost_per_ton
        )
        self.df['energy_cost'] = (
            self.df['energy_total_kWh'] * self.constraints.energy_cost_per_kwh
        )
        self.df['labor_cost'] = (
            self.df[self.PROCESS_TIME_COLUMNS].sum(axis=1) *
            self.constraints.labor_cost_per_hour
        )

    def _calculate_profit(self):
        """Calculate profit margin per plate."""
        self.df['profit'] = (
            self.df['price'] -
            (self.df['material_cost'] + self.df['energy_cost'] + self.df['labor_cost'])
        )

    def get_total_process_time(self, plate_id: str) -> float:
        """Get total process time for a specific plate."""
        row = self.df[self.df['id'] == plate_id]
        if row.empty:
            return 0.0
        return row[self.PROCESS_TIME_COLUMNS].sum(axis=1).values[0]


class PlateOptimizer:
    """
    Implements linear programming optimization for plate production scheduling.
    
    Uses PuLP to formulate and solve a multi-objective optimization problem
    that maximizes profit while considering KPI metrics.
    """

    def __init__(
        self,
        data_processor: PlateDataProcessor,
        weights: OptimizationWeights
    ):
        """
        Initialize optimizer with processed data and objective weights.
        
        Args:
            data_processor: PlateDataProcessor with processed plate data
            weights: OptimizationWeights for multi-objective optimization
        """
        self.data_processor = data_processor
        self.weights = weights
        self.model = None
        self.plate_vars = {}

    def optimize(self, constraints: OptimizationConstraints) -> Tuple[Dict[str, float], float]:
        """
        Formulate and solve the optimization problem.
        
        Args:
            constraints: OptimizationConstraints with shift time and plate limits
            
        Returns:
            Tuple of (decision_variables dict, objective_value)
        """
        df = self.data_processor.df
        
        # Create LP problem - maximization
        self.model = pulp.LpProblem("PlateProductionOptimizer", pulp.LpMaximize)

        # Decision variables: binary (schedule plate or not)
        self.plate_vars = {
            row['id']: pulp.LpVariable(f"x_{row['id']}", cat='Binary')
            for _, row in df.iterrows()
        }

        # Objective function: weighted sum of metrics
        objective = pulp.lpSum([
            self.plate_vars[row['id']] * self._calculate_plate_score(row)
            for _, row in df.iterrows()
        ])
        self.model += objective

        # Constraint 1: Total process time ≤ shift hours
        self.model += pulp.lpSum([
            self.plate_vars[row['id']] *
            self.data_processor.get_total_process_time(row['id'])
            for _, row in df.iterrows()
        ]) <= constraints.max_shift_hours

        # Constraint 2: Max plates per shift
        self.model += pulp.lpSum([
            self.plate_vars[row['id']]
            for _, row in df.iterrows()
        ]) <= constraints.max_plates_per_shift

        # Solve
        self.model.solve(pulp.PULP_CBC_CMD(msg=0))

        # Extract results
        results = {
            plate_id: float(var.varValue) if var.varValue is not None else 0.0
            for plate_id, var in self.plate_vars.items()
        }

        objective_value = float(self.model.objective.value()) if self.model.objective.value() else 0.0

        return results, objective_value

    def _calculate_plate_score(self, row: pd.Series) -> float:
        """
        Calculate weighted score for a plate.
        
        Args:
            row: Plate data row
            
        Returns:
            Weighted score combining profit and KPIs
        """
        return (
            self.weights.profit_weight * row['profit'] +
            self.weights.nesting_weight * row['nesting_utilization'] +
            self.weights.first_pass_weight * row['first_pass_yield'] +
            self.weights.energy_weight * row['energy_intensity_kWh_per_ton']
        )


class PlateOptimizationService:
    """
    High-level orchestrator for plate production optimization.
    
    Coordinates data processing, optimization, and result compilation.
    This is the main service class used by the API.
    """

    def __init__(self):
        """Initialize the optimization service."""
        self.data_processor = None
        self.optimizer = None

    def optimize_production(
        self,
        plates: List[PlateRequest],
        constraints: OptimizationConstraints,
        weights: OptimizationWeights
    ) -> OptimizationResponse:
        """
        Execute complete optimization workflow.
        
        Args:
            plates: List of available plates to schedule
            constraints: Production constraints
            weights: Objective function weights
            
        Returns:
            OptimizationResponse with scheduled plates and KPIs
        """
        try:
            # Process plate data
            self.data_processor = PlateDataProcessor(plates, constraints)
            df = self.data_processor.process_plates()

            # Run optimization
            self.optimizer = PlateOptimizer(self.data_processor, weights)
            results, obj_value = self.optimizer.optimize(constraints)

            # Extract scheduled plates
            scheduled_plate_ids = [
                pid for pid, value in results.items() if value == 1.0
            ]

            # Compile KPIs for scheduled plates
            scheduled_df = df[df['id'].isin(scheduled_plate_ids)]
            kpis = self._compile_kpis(scheduled_df)

            # Calculate totals
            total_profit = scheduled_df['profit'].sum()
            total_time = sum(
                self.data_processor.get_total_process_time(pid)
                for pid in scheduled_plate_ids
            )

            return OptimizationResponse(
                status="success",
                scheduled_plates=scheduled_plate_ids,
                kpis=kpis,
                total_profit=total_profit,
                total_process_time=total_time,
                optimization_value=obj_value
            )

        except Exception as e:
            return OptimizationResponse(
                status="failed",
                scheduled_plates=[],
                kpis=[],
                total_profit=0.0,
                total_process_time=0.0,
                optimization_value=0.0
            )

    def _compile_kpis(self, df: pd.DataFrame) -> List[PlateKPIResponse]:
        """
        Compile KPI responses from DataFrame.
        
        Args:
            df: DataFrame with scheduled plates
            
        Returns:
            List of PlateKPIResponse objects
        """
        kpis = []
        for _, row in df.iterrows():
            kpi = PlateKPIResponse(
                id=row['id'],
                profit=float(row['profit']),
                nesting_utilization=float(row['nesting_utilization']),
                first_pass_yield=float(row['first_pass_yield']),
                energy_intensity_kwh_per_ton=float(row['energy_intensity_kWh_per_ton']),
                material_cost=float(row['material_cost']),
                energy_cost=float(row['energy_cost']),
                labor_cost=float(row['labor_cost']),
                total_process_time=float(self.data_processor.get_total_process_time(row['id']))
            )
            kpis.append(kpi)
        return kpis
