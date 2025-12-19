import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import ResearchHub from './pages/ResearchHub';
import { LayoutDashboard, Library } from 'lucide-react';

const App: React.FC = () => {
    return (
        <Router>
            <div className="flex bg-slate-950 min-h-screen font-sans text-cyan-50">
                {/* Sidebar Navigation */}
                <nav className="w-64 border-r border-slate-800 p-6 flex flex-col gap-6 backdrop-blur-sm bg-slate-900/30 sticky top-0 h-screen">
                    <div className="text-2xl font-bold tracking-tighter text-cyan-400 mb-8">
                        NAVIGATOR
                    </div>

                    <Link to="/" className="flex items-center gap-3 p-3 rounded-lg hover:bg-cyan-900/20 text-slate-300 hover:text-cyan-400 transition-all">
                        <LayoutDashboard className="w-5 h-5" />
                        Dashboard
                    </Link>

                    <Link to="/research" className="flex items-center gap-3 p-3 rounded-lg hover:bg-cyan-900/20 text-slate-300 hover:text-cyan-400 transition-all">
                        <Library className="w-5 h-5" />
                        Research Hub
                    </Link>

                    <div className="mt-auto p-4 rounded-xl bg-gradient-to-br from-cyan-900/20 to-purple-900/20 border border-slate-800">
                        <div className="text-xs text-slate-400 uppercase tracking-wider mb-2">System Status</div>
                        <div className="flex items-center gap-2 text-green-400 text-sm font-bold">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                            </span>
                            TruthRL Active
                        </div>
                    </div>
                </nav>

                {/* Main Content Area */}
                <main className="flex-1 overflow-x-hidden">
                    <Routes>
                        <Route path="/" element={<Dashboard />} />
                        <Route path="/research" element={<ResearchHub />} />
                    </Routes>
                </main>
            </div>
        </Router>
    );
};

export default App;
