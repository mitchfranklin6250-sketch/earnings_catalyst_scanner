#!/usr/bin/env python3
"""
Earnings Catalyst Scanner - Enhanced Edition
Catches earnings beats, M&A targets, and multi-bagger opportunities
Target: 25-30 alerts per week across all tiers

Author: Mitch
Run Schedule: Daily at 6 AM, 12 PM, 6 PM ET (GitHub Actions)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
import requests
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')
MIN_MARKET_CAP = 250_000_000  # $250M - lowered to catch smaller movers
MAX_MARKET_CAP = 50_000_000_000  # $50B
MIN_AVG_VOLUME = 100_000  # Lower for small caps
LOOKBACK_DAYS = 14  # Scan earnings in next 14 days

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hot sectors for M&A and growth
HOT_SECTORS = {
    'Biotechnology': 1.5,
    'Drug Manufacturers - Specialty & Generic': 1.3,
    'Medical Devices': 1.2,
    'Diagnostics & Research': 1.4,
    'Semiconductors': 1.3,
    'Software - Application': 1.2,
    'Internet Content & Information': 1.2
}

HOT_KEYWORDS = [
    'obesity', 'glp-1', 'weight loss', 'ozempic', 'wegovy',
    'oncology', 'cancer', 'liquid biopsy', 'immunotherapy',
    'artificial intelligence', 'ai', 'machine learning',
    'cloud', 'saas', 'cybersecurity',
    'semiconductor', 'chip', 'gpu', 'ai accelerator'
]

class OpportunityScanner:
    """Main scanner class for earnings and M&A opportunities"""
    
    def __init__(self):
        self.opportunities = {
            'tier1': [],  # High conviction (score >= 8)
            'tier2': [],  # Strong (score 5-7)
            'tier3': []   # Watch (score 3-4)
        }
        self.scan_date = datetime.now()
    
    def get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 tickers as universe"""
        try:
            # Get S&P 500 list
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            
            # Clean tickers
            tickers = [t.replace('.', '-') for t in tickers]
            
            logger.info(f"Loaded {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.error(f"Error loading S&P 500 tickers: {e}")
            # Fallback to manual list of key tickers
            return self._get_fallback_tickers()
    
    def _get_fallback_tickers(self) -> List[str]:
        """Fallback ticker list if S&P 500 load fails"""
        return [
            # Healthcare/Biotech
            'GH', 'MTSR', 'EXAS', 'ILMN', 'VRTX', 'REGN', 'GILD', 'AMGN', 'BIIB',
            # Semiconductors
            'NVDA', 'AMD', 'AVGO', 'QCOM', 'MRVL', 'FORM', 'KLAC', 'LRCX', 'ASML',
            # Software/Cloud
            'CRM', 'NOW', 'SNOW', 'DDOG', 'CRWD', 'ZS', 'NET', 'PANW', 'OKTA',
            # Other growth
            'TSLA', 'SHOP', 'SQ', 'COIN', 'RBLX', 'U', 'DASH', 'ABNB'
        ]
    
    def get_upcoming_earnings(self, tickers: List[str]) -> List[Dict]:
        """Get tickers with earnings in next LOOKBACK_DAYS"""
        earnings_calendar = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                calendar = stock.calendar
                
                if calendar is None or calendar.empty:
                    continue
                
                next_earnings = calendar.get('Earnings Date')
                if next_earnings is None or len(next_earnings) == 0:
                    continue
                
                # Get first date (usually earliest estimate)
                earnings_date = next_earnings[0] if isinstance(next_earnings, list) else next_earnings
                
                if pd.isna(earnings_date):
                    continue
                
                days_to_earnings = (earnings_date - pd.Timestamp.now()).days
                
                # Only include if within lookback window
                if 0 <= days_to_earnings <= LOOKBACK_DAYS:
                    earnings_calendar.append({
                        'ticker': ticker,
                        'earnings_date': earnings_date,
                        'days_to_earnings': days_to_earnings
                    })
                    
            except Exception as e:
                logger.debug(f"Error getting calendar for {ticker}: {e}")
                continue
        
        logger.info(f"Found {len(earnings_calendar)} tickers with upcoming earnings")
        return earnings_calendar
    
    def screen_basic_criteria(self, ticker: str) -> Tuple[bool, Dict]:
        """Screen for basic market cap, volume, momentum criteria"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period='3mo')
            
            if hist.empty or len(hist) < 20:
                return False, {}
            
            # Basic metrics
            market_cap = info.get('marketCap', 0)
            avg_volume = info.get('averageVolume', 0)
            current_price = hist['Close'].iloc[-1]
            
            # Market cap filter
            if not (MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP):
                return False, {}
            
            # Volume filter (adjust for market cap)
            if market_cap < 1_000_000_000:
                min_vol = MIN_AVG_VOLUME
            else:
                min_vol = 250_000
            
            if avg_volume < min_vol:
                return False, {}
            
            # Price momentum - should be within 20% of 52-week high
            week_52_high = hist['High'].tail(252).max() if len(hist) >= 252 else hist['High'].max()
            pct_from_high = (week_52_high - current_price) / week_52_high
            
            if pct_from_high > 0.20:  # More than 20% from high
                return False, {}
            
            # Recent momentum - up in last 20 days
            if len(hist) >= 20:
                price_20d_ago = hist['Close'].iloc[-20]
                recent_return = (current_price - price_20d_ago) / price_20d_ago
            else:
                recent_return = 0
            
            metrics = {
                'market_cap': market_cap,
                'avg_volume': avg_volume,
                'current_price': current_price,
                'pct_from_high': pct_from_high,
                'recent_return': recent_return
            }
            
            return True, metrics
            
        except Exception as e:
            logger.debug(f"Error in basic screen for {ticker}: {e}")
            return False, {}
    
    def get_earnings_history_score(self, ticker: str) -> Dict:
        """Calculate earnings beat history and consistency"""
        try:
            stock = yf.Ticker(ticker)
            earnings_dates = stock.earnings_dates
            
            if earnings_dates is None or len(earnings_dates) < 4:
                return {'beat_rate': 0, 'avg_surprise': 0, 'consistency': 0}
            
            # Get last 8 quarters
            recent_earnings = earnings_dates.head(8)
            
            beats = 0
            total = 0
            surprises = []
            
            for idx, row in recent_earnings.iterrows():
                if 'Surprise(%)' in row and pd.notna(row['Surprise(%)']):
                    total += 1
                    surprise_pct = row['Surprise(%)']
                    surprises.append(surprise_pct)
                    
                    if surprise_pct > 0:
                        beats += 1
            
            if total == 0:
                return {'beat_rate': 0, 'avg_surprise': 0, 'consistency': 0}
            
            beat_rate = beats / total
            avg_surprise = np.mean(surprises)
            
            # Consistency: standard deviation of surprises (lower is more consistent)
            consistency = 1 - min(np.std(surprises) / 20, 1.0) if len(surprises) > 2 else 0.5
            
            return {
                'beat_rate': beat_rate,
                'avg_surprise': avg_surprise,
                'consistency': consistency,
                'total_quarters': total
            }
            
        except Exception as e:
            logger.debug(f"Error getting earnings history for {ticker}: {e}")
            return {'beat_rate': 0, 'avg_surprise': 0, 'consistency': 0}
    
    def get_revenue_growth_score(self, ticker: str) -> Dict:
        """Check for revenue acceleration (key for stocks like GH, FORM)"""
        try:
            stock = yf.Ticker(ticker)
            quarterly_financials = stock.quarterly_financials
            
            if quarterly_financials is None or len(quarterly_financials.columns) < 3:
                return {'qoq_growth': 0, 'yoy_growth': 0, 'acceleration': False}
            
            # Get total revenue row
            if 'Total Revenue' not in quarterly_financials.index:
                return {'qoq_growth': 0, 'yoy_growth': 0, 'acceleration': False}
            
            revenue_data = quarterly_financials.loc['Total Revenue']
            
            # Most recent 3 quarters
            if len(revenue_data) >= 3:
                q1 = revenue_data.iloc[0]  # Most recent
                q2 = revenue_data.iloc[1]  # Previous quarter
                q3 = revenue_data.iloc[2]  # 2 quarters ago
                
                if pd.notna(q1) and pd.notna(q2) and q2 > 0:
                    qoq_growth = (q1 - q2) / q2
                else:
                    qoq_growth = 0
                
                # Check if accelerating
                if pd.notna(q2) and pd.notna(q3) and q3 > 0:
                    prev_qoq_growth = (q2 - q3) / q3
                    acceleration = qoq_growth > prev_qoq_growth
                else:
                    acceleration = False
            else:
                qoq_growth = 0
                acceleration = False
            
            # YoY growth (q1 vs q5 if available)
            if len(revenue_data) >= 5:
                q5 = revenue_data.iloc[4]
                if pd.notna(q1) and pd.notna(q5) and q5 > 0:
                    yoy_growth = (q1 - q5) / q5
                else:
                    yoy_growth = 0
            else:
                yoy_growth = 0
            
            return {
                'qoq_growth': qoq_growth,
                'yoy_growth': yoy_growth,
                'acceleration': acceleration
            }
            
        except Exception as e:
            logger.debug(f"Error getting revenue growth for {ticker}: {e}")
            return {'qoq_growth': 0, 'yoy_growth': 0, 'acceleration': False}
    
    def get_sector_and_keyword_score(self, ticker: str) -> Dict:
        """Score based on hot sectors and keywords"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            industry = info.get('industry', '')
            sector = info.get('sector', '')
            business_summary = info.get('longBusinessSummary', '').lower()
            
            # Sector multiplier
            sector_multiplier = HOT_SECTORS.get(industry, 1.0)
            
            # Keyword matches
            keyword_matches = [kw for kw in HOT_KEYWORDS if kw in business_summary]
            keyword_score = len(keyword_matches) * 0.5
            
            return {
                'sector': sector,
                'industry': industry,
                'sector_multiplier': sector_multiplier,
                'keyword_matches': keyword_matches,
                'keyword_score': keyword_score
            }
            
        except Exception as e:
            logger.debug(f"Error getting sector info for {ticker}: {e}")
            return {
                'sector': '',
                'industry': '',
                'sector_multiplier': 1.0,
                'keyword_matches': [],
                'keyword_score': 0
            }
    
    def get_options_signals(self, ticker: str) -> Dict:
        """Get options-based signals (call/put ratio, IV)"""
        try:
            stock = yf.Ticker(ticker)
            options_dates = stock.options
            
            if len(options_dates) == 0:
                return {
                    'call_put_ratio': 0,
                    'unusual_activity': False,
                    'high_iv': False
                }
            
            # Get nearest expiration
            nearest_exp = options_dates[0]
            opt_chain = stock.option_chain(nearest_exp)
            
            # Calculate call/put volume ratio
            calls_vol = opt_chain.calls['volume'].fillna(0).sum()
            puts_vol = opt_chain.puts['volume'].fillna(0).sum()
            
            if puts_vol > 0:
                call_put_ratio = calls_vol / puts_vol
            else:
                call_put_ratio = 0
            
            # Check for unusual activity
            # (Call ratio > 2.5 suggests bullish positioning)
            unusual_activity = call_put_ratio > 2.5
            
            # Check implied volatility
            # (High IV suggests big move expected)
            avg_iv = opt_chain.calls['impliedVolatility'].mean()
            high_iv = avg_iv > 0.60 if pd.notna(avg_iv) else False
            
            return {
                'call_put_ratio': call_put_ratio,
                'unusual_activity': unusual_activity,
                'high_iv': high_iv
            }
            
        except Exception as e:
            logger.debug(f"Error getting options signals for {ticker}: {e}")
            return {
                'call_put_ratio': 0,
                'unusual_activity': False,
                'high_iv': False
            }
    
    def get_short_interest_score(self, ticker: str) -> float:
        """Get short interest for squeeze potential"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            short_percent = info.get('shortPercentOfFloat', 0)
            
            if short_percent is None:
                return 0
            
            # Convert to float if needed
            if isinstance(short_percent, str):
                short_percent = float(short_percent.strip('%')) / 100
            
            return short_percent
            
        except Exception as e:
            logger.debug(f"Error getting short interest for {ticker}: {e}")
            return 0
    
    def calculate_composite_score(self, ticker: str, earnings_data: Dict, 
                                  basic_metrics: Dict) -> Dict:
        """Calculate composite opportunity score"""
        
        # Get all components
        earnings_history = self.get_earnings_history_score(ticker)
        revenue_growth = self.get_revenue_growth_score(ticker)
        sector_info = self.get_sector_and_keyword_score(ticker)
        options_signals = self.get_options_signals(ticker)
        short_interest = self.get_short_interest_score(ticker)
        
        # Scoring components (max 10 points)
        score = 0
        score_breakdown = {}
        
        # 1. Earnings beat history (0-2.5 points)
        beat_score = earnings_history['beat_rate'] * 2.5
        score += beat_score
        score_breakdown['earnings_beat'] = round(beat_score, 2)
        
        # 2. Revenue growth (0-2 points)
        revenue_score = 0
        if revenue_growth['acceleration']:
            revenue_score += 1.0
        if revenue_growth['qoq_growth'] > 0.05:  # >5% QoQ
            revenue_score += 0.5
        if revenue_growth['yoy_growth'] > 0.20:  # >20% YoY
            revenue_score += 0.5
        score += revenue_score
        score_breakdown['revenue_growth'] = round(revenue_score, 2)
        
        # 3. Sector/keyword relevance (0-2 points)
        sector_score = min((sector_info['sector_multiplier'] - 1.0) * 2, 1.0)
        keyword_score = min(sector_info['keyword_score'], 1.0)
        combined_sector = sector_score + keyword_score
        score += combined_sector
        score_breakdown['sector_keywords'] = round(combined_sector, 2)
        
        # 4. Options signals (0-1.5 points)
        options_score = 0
        if options_signals['unusual_activity']:
            options_score += 0.75
        if options_signals['high_iv']:
            options_score += 0.75
        score += options_score
        score_breakdown['options'] = round(options_score, 2)
        
        # 5. Short squeeze potential (0-1 point)
        short_score = min(short_interest / 0.20, 1.0)  # Max at 20% SI
        score += short_score
        score_breakdown['short_squeeze'] = round(short_score, 2)
        
        # 6. Price momentum (0-1 point)
        momentum_score = 0
        if basic_metrics['pct_from_high'] < 0.10:  # Within 10% of high
            momentum_score += 0.5
        if basic_metrics['recent_return'] > 0:
            momentum_score += 0.5
        score += momentum_score
        score_breakdown['momentum'] = round(momentum_score, 2)
        
        # Total score
        total_score = round(score, 2)
        
        return {
            'ticker': ticker,
            'total_score': total_score,
            'score_breakdown': score_breakdown,
            'days_to_earnings': earnings_data['days_to_earnings'],
            'earnings_date': earnings_data['earnings_date'].strftime('%Y-%m-%d'),
            'market_cap': basic_metrics['market_cap'],
            'current_price': basic_metrics['current_price'],
            'earnings_beat_rate': earnings_history['beat_rate'],
            'qoq_growth': revenue_growth['qoq_growth'],
            'sector': sector_info['sector'],
            'industry': sector_info['industry'],
            'keyword_matches': sector_info['keyword_matches'],
            'call_put_ratio': options_signals['call_put_ratio'],
            'short_interest': short_interest,
            'pct_from_high': basic_metrics['pct_from_high']
        }
    
    def scan_opportunities(self):
        """Main scanning function"""
        logger.info("Starting earnings catalyst scan...")
        
        # Get universe
        tickers = self.get_sp500_tickers()
        
        # Get upcoming earnings
        earnings_calendar = self.get_upcoming_earnings(tickers)
        
        if len(earnings_calendar) == 0:
            logger.warning("No upcoming earnings found in calendar")
            return
        
        logger.info(f"Scanning {len(earnings_calendar)} tickers with upcoming earnings...")
        
        all_opportunities = []
        
        for earnings_data in earnings_calendar:
            ticker = earnings_data['ticker']
            
            # Basic screen
            passes_screen, basic_metrics = self.screen_basic_criteria(ticker)
            
            if not passes_screen:
                continue
            
            # Calculate composite score
            try:
                opportunity = self.calculate_composite_score(ticker, earnings_data, basic_metrics)
                all_opportunities.append(opportunity)
                logger.info(f"Scored {ticker}: {opportunity['total_score']}/10")
            except Exception as e:
                logger.error(f"Error scoring {ticker}: {e}")
                continue
        
        # Sort by score
        all_opportunities.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Categorize into tiers
        for opp in all_opportunities:
            score = opp['total_score']
            if score >= 7.0:
                self.opportunities['tier1'].append(opp)
            elif score >= 5.0:
                self.opportunities['tier2'].append(opp)
            elif score >= 3.0:
                self.opportunities['tier3'].append(opp)
        
        logger.info(f"Scan complete: T1={len(self.opportunities['tier1'])}, "
                   f"T2={len(self.opportunities['tier2'])}, "
                   f"T3={len(self.opportunities['tier3'])}")
    
    def format_opportunity_message(self, opp: Dict, detailed: bool = True) -> str:
        """Format opportunity for Discord message"""
        ticker = opp['ticker']
        score = opp['total_score']
        
        msg = f"**${ticker}** - Score: {score}/10\n"
        msg += f"├─ Earnings: {opp['earnings_date']} ({opp['days_to_earnings']} days)\n"
        msg += f"├─ Price: ${opp['current_price']:.2f}\n"
        msg += f"├─ Market Cap: ${opp['market_cap']/1e9:.2f}B\n"
        
        if detailed:
            msg += f"├─ Beat Rate: {opp['earnings_beat_rate']:.1%}\n"
            msg += f"├─ QoQ Growth: {opp['qoq_growth']:.1%}\n"
            msg += f"├─ Call/Put: {opp['call_put_ratio']:.2f}\n"
            msg += f"├─ Short Interest: {opp['short_interest']:.1%}\n"
            msg += f"├─ From High: {opp['pct_from_high']:.1%}\n"
            
            if len(opp['keyword_matches']) > 0:
                keywords = ', '.join(opp['keyword_matches'][:3])
                msg += f"├─ Keywords: {keywords}\n"
            
            # Show score breakdown
            breakdown = opp['score_breakdown']
            top_factors = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:2]
            factors_str = ', '.join([f"{k.replace('_', ' ').title()}: {v}" 
                                    for k, v in top_factors if v > 0])
            msg += f"└─ Top Factors: {factors_str}\n\n"
        else:
            msg += f"└─ Beat Rate: {opp['earnings_beat_rate']:.1%}\n\n"
        
        return msg
    
    def send_discord_alert(self):
        """Send tiered alerts to Discord"""
        
        if not DISCORD_WEBHOOK_URL:
            logger.warning("Discord webhook not configured")
            return
        
        # Tier 1: High Conviction
        if len(self.opportunities['tier1']) > 0:
            msg = "🚨 **TIER 1: HIGH CONVICTION EARNINGS PLAYS** 🚨\n"
            msg += f"*{len(self.opportunities['tier1'])} opportunities with score ≥ 7.0*\n\n"
            
            for opp in self.opportunities['tier1'][:10]:  # Max 10
                msg += self.format_opportunity_message(opp, detailed=True)
            
            self._send_webhook(msg, urgent=True)
        
        # Tier 2: Strong
        if len(self.opportunities['tier2']) > 0:
            msg = "⚡ **TIER 2: STRONG EARNINGS PLAYS**\n"
            msg += f"*{len(self.opportunities['tier2'])} opportunities with score 5.0-6.9*\n\n"
            
            for opp in self.opportunities['tier2'][:15]:  # Max 15
                msg += self.format_opportunity_message(opp, detailed=False)
            
            self._send_webhook(msg)
        
        # Tier 3: Watch List (compact format)
        if len(self.opportunities['tier3']) > 0:
            msg = "📊 **TIER 3: WATCH LIST**\n"
            msg += f"*{len(self.opportunities['tier3'])} opportunities with score 3.0-4.9*\n\n"
            
            # Group by days to earnings
            by_date = {}
            for opp in self.opportunities['tier3']:
                date = opp['earnings_date']
                if date not in by_date:
                    by_date[date] = []
                by_date[date].append(opp)
            
            for date in sorted(by_date.keys()):
                opps = by_date[date]
                tickers = [f"${o['ticker']}" for o in opps[:10]]
                msg += f"**{date}**: {', '.join(tickers)}\n"
            
            self._send_webhook(msg)
        
        # Summary stats
        total = sum(len(v) for v in self.opportunities.values())
        summary = f"\n📈 **SCAN SUMMARY** - {self.scan_date.strftime('%Y-%m-%d %H:%M ET')}\n"
        summary += f"Total Opportunities: {total}\n"
        summary += f"├─ Tier 1 (High): {len(self.opportunities['tier1'])}\n"
        summary += f"├─ Tier 2 (Strong): {len(self.opportunities['tier2'])}\n"
        summary += f"└─ Tier 3 (Watch): {len(self.opportunities['tier3'])}\n"
        
        self._send_webhook(summary)
    
    def _send_webhook(self, message: str, urgent: bool = False):
        """Send message to Discord webhook"""
        try:
            # Split long messages
            max_length = 1900
            if len(message) > max_length:
                parts = [message[i:i+max_length] 
                        for i in range(0, len(message), max_length)]
            else:
                parts = [message]
            
            for part in parts:
                payload = {
                    'content': part,
                    'username': 'Earnings Catalyst Scanner' if not urgent else '🚨 HIGH CONVICTION ALERT'
                }
                
                response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
                
                if response.status_code != 204:
                    logger.error(f"Discord webhook error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")
    
    def save_results(self):
        """Save scan results to JSON file"""
        try:
            results = {
                'scan_date': self.scan_date.isoformat(),
                'opportunities': self.opportunities
            }
            
            filename = f"scan_results_{self.scan_date.strftime('%Y%m%d_%H%M')}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info("EARNINGS CATALYST SCANNER - ENHANCED EDITION")
    logger.info("="*60)
    
    scanner = OpportunityScanner()
    
    # Run the scan
    scanner.scan_opportunities()
    
    # Send alerts
    scanner.send_discord_alert()
    
    # Save results
    scanner.save_results()
    
    logger.info("Scan complete!")


if __name__ == "__main__":
    main()
