import { redirect } from 'next/navigation';
import Link from 'next/link';
import { createClient } from '@/lib/supabase/server';
import { Button } from '@/components/ui/button';
import { LogoutButton } from './logout-button';

export default async function ProfilePage() {
  const supabase = await createClient();

  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect('/auth/login');
  }

  // Fetch profile and stats
  const { data: profile } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', user.id)
    .single();

  const { data: stats } = await supabase
    .from('player_stats')
    .select('*')
    .eq('user_id', user.id)
    .single();

  const winRate = stats && stats.games_played > 0
    ? ((stats.games_won / stats.games_played) * 100).toFixed(1)
    : '0.0';

  return (
    <div className="min-h-screen bg-gray-100 py-12">
      <div className="max-w-2xl mx-auto px-4">
        {/* Profile Header */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center text-white text-2xl font-bold">
                {profile?.display_name?.charAt(0).toUpperCase() || 'P'}
              </div>
              <div>
                <h1 className="text-2xl font-bold">{profile?.display_name || 'Player'}</h1>
                <p className="text-gray-500">@{profile?.username || 'unknown'}</p>
              </div>
            </div>
            <LogoutButton />
          </div>

          <div className="text-sm text-gray-500">
            Member since {profile?.created_at ? new Date(profile.created_at).toLocaleDateString() : 'Unknown'}
          </div>
        </div>

        {/* Stats Card */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">Statistics</h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Games Played" value={stats?.games_played || 0} />
            <StatCard label="Games Won" value={stats?.games_won || 0} />
            <StatCard label="Win Rate" value={`${winRate}%`} />
            <StatCard label="ELO Rating" value={stats?.elo_rating || 1000} highlight />
          </div>

          <div className="grid grid-cols-2 gap-4 mt-4">
            <StatCard label="Highest ELO" value={stats?.highest_elo || 1000} />
            <StatCard label="Best Win Streak" value={stats?.best_win_streak || 0} />
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold mb-4">Play</h2>
          <div className="flex flex-wrap gap-3">
            <Link href="/play/bot">
              <Button>Play vs Bots</Button>
            </Link>
            <Link href="/play/online">
              <Button variant="outline">Play Online</Button>
            </Link>
            <Link href="/play/test">
              <Button variant="outline">Test Mode</Button>
            </Link>
          </div>
        </div>

        {/* Back Link */}
        <div className="mt-6 text-center">
          <Link href="/" className="text-gray-500 hover:underline">
            Back to home
          </Link>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string | number;
  highlight?: boolean;
}) {
  return (
    <div className={`p-4 rounded-lg ${highlight ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50'}`}>
      <div className="text-sm text-gray-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${highlight ? 'text-blue-600' : ''}`}>{value}</div>
    </div>
  );
}
