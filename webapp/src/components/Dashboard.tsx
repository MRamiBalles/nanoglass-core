import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, Shield, Zap, AlertTriangle, Terminal } from 'lucide-react';

const mockEnergyData = [
    { step: 10, energy: 0.95, entropy: 0.8 },
    { step: 20, energy: 0.82, entropy: 0.75 },
    { step: 30, energy: 0.65, entropy: 0.6 },
    { step: 40, energy: 0.45, entropy: 0.55 },
    { step: 50, energy: 0.32, entropy: 0.4 },
    { step: 60, energy: 0.21, entropy: 0.35 },
    { step: 70, energy: 0.15, entropy: 0.2 },
];

const Dashboard: React.FC = () => {
    return (
        <div className="min-h-screen bg-slate-950 text-cyan-50 p-8 font-mono relative overflow-hidden">
            {/* Background Glow */}
            <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(circle_at_50%_50%,_rgba(15,23,42,0.9),_rgba(2,6,23,1))] -z-10"></div>
            <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-cyan-900/20 blur-[100px] rounded-full"></div>

            <header className="mb-10 flex justify-between items-center border-b border-cyan-900/50 pb-4">
                <div>
                    <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-400">
                        PROJECT NANOGLASS
                    </h1>
                    <p className="text-cyan-400/60 mt-2 text-sm tracking-widest">GLASS BOX INTERPRETER V1.0</p>
                </div>
                <div className="flex gap-4">
                    <div className="px-4 py-2 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                        SYSTEM ONLINE
                    </div>
                </div>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                {/* Metric Cards */}
                <div className="p-6 rounded-2xl bg-slate-900/50 backdrop-blur-md border border-cyan-800/30 hover:border-cyan-500/50 transition-all group">
                    <div className="flex justify-between items-start mb-4">
                        <h3 className="text-cyan-400/80 text-sm">CURRENT ENERGY (L1)</h3>
                        <Zap className="w-5 h-5 text-yellow-400 group-hover:scale-110 transition-transform" />
                    </div>
                    <div className="text-4xl font-bold text-white mb-2">0.1542</div>
                    <div className="text-xs text-green-400">↓ Minimizing (Optimal)</div>
                </div>

                <div className="p-6 rounded-2xl bg-slate-900/50 backdrop-blur-md border border-cyan-800/30 hover:border-cyan-500/50 transition-all group">
                    <div className="flex justify-between items-start mb-4">
                        <h3 className="text-purple-400/80 text-sm">ENTROPY STATE</h3>
                        <Activity className="w-5 h-5 text-purple-400 group-hover:scale-110 transition-transform" />
                    </div>
                    <div className="text-4xl font-bold text-white mb-2">Low</div>
                    <div className="text-xs text-purple-300/60">Stable Geometry</div>
                </div>

                <div className="p-6 rounded-2xl bg-slate-900/50 backdrop-blur-md border border-cyan-800/30 hover:border-cyan-500/50 transition-all group">
                    <div className="flex justify-between items-start mb-4">
                        <h3 className="text-green-400/80 text-sm">TruthRL STATUS</h3>
                        <Shield className="w-5 h-5 text-green-400 group-hover:scale-110 transition-transform" />
                    </div>
                    <div className="text-4xl font-bold text-white mb-2">Active</div>
                    <div className="text-xs text-cyan-300/60">Abstention Rate: 12%</div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Main Chart */}
                <div className="lg:col-span-2 p-6 rounded-2xl bg-slate-900/50 backdrop-blur-md border border-cyan-800/30">
                    <h3 className="text-xl font-semibold mb-6 text-cyan-100 flex items-center gap-2">
                        <Activity className="w-5 h-5" /> Thermodynamics of Meaning
                    </h3>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={mockEnergyData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                <XAxis dataKey="step" stroke="#94a3b8" />
                                <YAxis stroke="#94a3b8" />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }}
                                    itemStyle={{ color: '#22d3ee' }}
                                />
                                <Line type="monotone" dataKey="energy" stroke="#22d3ee" strokeWidth={3} dot={{ r: 4, fill: '#22d3ee' }} activeDot={{ r: 8 }} />
                                <Line type="monotone" dataKey="entropy" stroke="#c084fc" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Live Terminal Log */}
                <div className="p-6 rounded-2xl bg-black/80 backdrop-blur-md border border-cyan-800/30 font-mono text-sm overflow-hidden flex flex-col">
                    <h3 className="text-sm font-semibold mb-4 text-cyan-400/80 flex items-center gap-2">
                        <Terminal className="w-4 h-4" /> LIVE INFERENCE LOG
                    </h3>
                    <div className="flex-1 overflow-y-auto space-y-3 scrollbar-hide">
                        <div className="p-2 border-l-2 border-green-500 bg-green-500/5 text-green-300">
                            <span className="opacity-50">[10:42:01]</span> Input: "What is 2+2?" <br />
                            <span className="text-white">Output: "4" (Energy: 0.12)</span>
                        </div>
                        <div className="p-2 border-l-2 border-yellow-500 bg-yellow-500/5 text-yellow-300">
                            <span className="opacity-50">[10:42:05]</span> Input: "Meaning of life?" <br />
                            <span className="text-white">Output: [IDK] (Energy: 0.89 - High Entropy)</span>
                        </div>
                        <div className="p-2 border-l-2 border-red-500 bg-red-500/5 text-red-300">
                            <span className="opacity-50">[10:42:15]</span> <AlertTriangle className="inline w-3 h-3 mr-1" /> Hallucination Attempt Blocked <br />
                            <span className="text-white/60">Reason: TruthRL Penalty > 2.0</span>
                        </div>
                        <div className="p-2 border-l-2 border-cyan-500 bg-cyan-500/5 text-cyan-300 animate-pulse">
                            <span className="opacity-50">[10:42:20]</span> Processing...
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
